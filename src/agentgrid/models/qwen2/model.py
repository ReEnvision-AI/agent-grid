import os
from typing import Union
from hivemind import DHT, get_logger

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import is_torch_flex_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2ForSequenceClassification, Qwen2Model, Qwen2PreTrainedModel

from agentgrid.client.from_pretrained import FromPretrainedMixin
from agentgrid.client.lm_head import LMHead
from agentgrid.client.ptune import PTuneMixin
from agentgrid.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from agentgrid.client.remote_sequential import RemoteSequential
from agentgrid.models.qwen2.config import DistributedQwen2Config
from agentgrid.models._mask_cache import LRUMaskCache

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

logger = get_logger(__name__)

class DistributedQwen2Model(FromPretrainedMixin, PTuneMixin, Qwen2Model):

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedQwen2Config

    def __init__(self, config: DistributedQwen2Config, *, dht: DHT | None = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0 # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)
        self.init_prompts(config)
        cache_size = int(os.getenv("AGENTGRID_MASK_CACHE_SIZE", "32"))
        self._mask_cache = LRUMaskCache(max_size=cache_size)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: RemotePastKeyValues | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
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

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }

        can_cache_mask = (
            attention_mask is None
            and (past_key_values is None or past_key_values.get_seq_length() == 0)
            and cache_position.numel() > 0
            and cache_position.min().item() == 0
        )

        causal_mask = None
        if can_cache_mask:
            cache_key = (inputs_embeds.shape[1], inputs_embeds.device.type, inputs_embeds.device.index)
            cached = self._mask_cache.get(cache_key)
            if cached is not None:
                causal_mask = cached
        if causal_mask is None:
            causal_mask = create_causal_mask(**mask_kwargs)
            if causal_mask is not None and causal_mask.device != inputs_embeds.device:
                causal_mask = causal_mask.to(inputs_embeds.device)
            if can_cache_mask and causal_mask is not None:
                self._mask_cache.put(cache_key, causal_mask)

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

    
class DistributedQwen2ForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, Qwen2ForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedQwen2Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedQwen2Model._keys_to_ignore_on_load_unexpected

    config_class = DistributedQwen2Config

    def __init__(self, config: DistributedQwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = DistributedQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head
    
    @property
    def transformer(self) -> DistributedQwen2Model:  # For compatibility with RemoteGenerationMixin
        return self.model

class DistributedQwen2ForSequenceClassification(FromPretrainedMixin, Qwen2ForSequenceClassification):
    _keys_to_ignore_on_load_missing = DistributedQwen2Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedQwen2Model._keys_to_ignore_on_load_unexpected

    config_class = DistributedQwen2Config

    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.model = DistributedQwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.post_init()

    @property
    def transformer(self) -> DistributedQwen2Model:
        return self.model
