from typing import Optional

import torch
import torch.nn as nn
from hivemind import DHT
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
    GptOssPreTrainedModel,
)

from agentgrid.client.from_pretrained import FromPretrainedMixin
from agentgrid.client.lm_head import LMHead
from agentgrid.client.ptune import PTuneMixin
from agentgrid.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from agentgrid.client.remote_sequential import RemoteSequential
from agentgrid.models.gpt_oss.config import DistributedGptOssConfig


class DistributedGptOssModel(FromPretrainedMixin, PTuneMixin, GptOssModel):
    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]
    _keep_in_fp32_modules = []

    config_class = DistributedGptOssConfig

    def __init__(self, config: DistributedGptOssConfig, *, dht: Optional[DHT] = None):
        n_layer = config.num_hidden_layers
        config.num_hidden_layers = 0
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)
        self.requires_grad_(False)
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
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = None,
    ) -> MoeModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if output_attentions:
            raise ValueError("DistributedGptOssModel does not support output_attentions=True")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and past_key_values is None:
            past_key_values = RemotePastKeyValues()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if output_router_logits:
            raise ValueError("DistributedGptOssModel does not support output_router_logits=True")

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
        }
        full_mask_kwargs = mask_kwargs.copy()
        full_mask_kwargs["past_key_values"] = None
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**full_mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**full_mask_kwargs),
        }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = (hidden_states,) if output_hidden_states else None

        hidden_states = self.layers(
            hidden_states,
            prompts=None,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        if past_key_values is not None:
            past_key_values.update_seen(hidden_states.size(1))

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
            router_logits=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:
        return self.layers

    @property
    def ln_f(self) -> nn.Module:
        return self.norm


class DistributedGptOssForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, GptOssForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedGptOssModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedGptOssModel._keys_to_ignore_on_load_unexpected
    _keep_in_fp32_modules = []

    config_class = DistributedGptOssConfig

    def __init__(self, config: DistributedGptOssConfig):
        GptOssPreTrainedModel.__init__(self, config)
        self.model = DistributedGptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedGptOssModel:
        return self.model
