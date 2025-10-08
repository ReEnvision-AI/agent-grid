#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from typing import Optional, Union

import hivemind
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import is_torch_flex_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel
from transformers.masking_utils import create_causal_mask

from agentgrid.client.from_pretrained import FromPretrainedMixin
from agentgrid.client.lm_head import LMHead
from agentgrid.client.ptune import PTuneMixin
from agentgrid.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from agentgrid.client.remote_sequential import RemoteSequential
from agentgrid.models.llama.config import DistributedLlamaConfig

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

logger = get_logger(__name__)


class DistributedLlamaModel(FromPretrainedMixin, PTuneMixin, LlamaModel):
    """LlamaModel, but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedLlamaConfig

    def __init__(self, config: DistributedLlamaConfig, *, dht: Optional[hivemind.DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings and layernorm
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = RemotePastKeyValues()


        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = (hidden_states, ) if output_hidden_states else None


        hidden_states = self.layers(
            hidden_states,
            prompts=None,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        past_key_values.update_seen(hidden_states.size(1))


        # Add last hidden state
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
        )


    @property
    def word_embeddings(self) -> nn.Embedding:  # For compatibility with RemoteGenerationMixin
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:  # For compatibility with RemoteGenerationMixin
        return self.layers

    @property
    def ln_f(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return self.norm


class DistributedLlamaForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, LlamaForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedLlamaModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedLlamaModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedLlamaConfig

    def __init__(self, config: DistributedLlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = DistributedLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedLlamaModel:  # For compatibility with RemoteGenerationMixin
        return self.model


class DistributedLlamaForSequenceClassification(FromPretrainedMixin, LlamaForSequenceClassification):
    _keys_to_ignore_on_load_missing = DistributedLlamaModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedLlamaModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedLlamaConfig

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.model = DistributedLlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def transformer(self) -> DistributedLlamaModel:  # For compatibility with RemoteGenerationMixin
        return self.model
