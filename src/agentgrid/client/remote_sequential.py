from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from hivemind import DHT, get_logger
from torch import nn

from agentgrid.client.config import ClientConfig
from agentgrid.client.inference_session import InferenceSession
from agentgrid.client.performance_monitor import timed_operation
from agentgrid.client.routing import RemoteSequenceManager
from agentgrid.client.sequential_autograd import _RemoteSequentialAutogradFunction
from agentgrid.client.session_pool import CachedSequenceManager, get_session_pool
from agentgrid.data_structures import UID_DELIMITER

logger = get_logger(__name__)


class RemoteSequential(nn.Module):
    """
    A sequence of transformer blocks hosted by the swarm.
    """

    def __init__(
        self,
        config: ClientConfig,
        *,
        sequence_manager: RemoteSequenceManager | None = None,
        dht: DHT | None = None,
        start_block: int | None = None,
        end_block: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        assert sequence_manager is None or (
            dht is None and start_block is None and end_block is None
        ), "`dht`, `start_block`, and `end_block` have no effect when you provide a custom `sequence_manager`"
        if sequence_manager is None:
            if start_block is None:
                start_block = 0
            if end_block is None:
                end_block = self.config.num_hidden_layers
            block_uids = tuple(f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(start_block, end_block))
            sequence_manager = RemoteSequenceManager(config, block_uids, dht=dht, **kwargs)

        # Wrap with caching layer to reduce overhead
        self.sequence_manager = CachedSequenceManager(sequence_manager)
        self._session_pool = get_session_pool()

        self._active_session = ContextVar("active_session", default=None)

    def forward(self, inputs: torch.Tensor, prompts: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, **kwargs) -> torch.Tensor:
        assert inputs.ndim == 3, "inputs must be a tensor of shape [batch_size, seq_length, hidden_size]"
        if self.active_session is None:
            return _RemoteSequentialAutogradFunction.apply(inputs, prompts, attention_mask, position_ids, position_embeddings, self.sequence_manager, kwargs)
        else:
            return self.active_session.step(inputs, prompts, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings, **kwargs)

    @property
    def active_session(self) -> InferenceSession | None:
        """
        If called inside `with model.inference_session(...):` or `with model.use_session(...):`,
        returns an active InferenceSession. Otherwise, returns None.
        """

        return self._active_session.get()

    @property
    def position(self) -> int:
        """Returns the prefix length (in tokens) in the active inference session or zero if no session is active."""

        return self.active_session.position if self.active_session is not None else 0

    @contextmanager
    def use_session(self, session: InferenceSession | None) -> Generator[InferenceSession, None, None]:
        """Inside this context, forward() will use an _existing_ InferenceSession provided as the argument."""

        token = self._active_session.set(session)
        try:
            yield session
        finally:
            self._active_session.reset(token)

    @contextmanager
    def inference_session(self, **kwargs) -> Generator[InferenceSession, None, None]:
        """
        Inside this context, forward() will use a _new_ InferenceSession created with given parameters.

        :param max_length: Maximal expected length of inference results. Servers use this parameter
                           to calculate the size of attention caches allocated to this client.
        """
        max_length = kwargs.get('max_length', 2048)

        with timed_operation("inference_session_creation"):
            # Always try to get spans from cache first to avoid redundant computation
            spans = None
            try:
                with timed_operation("span_computation"):
                    spans = self.sequence_manager.make_sequence_cached(
                        mode="min_latency",
                        cache_tokens_needed=max_length
                    )
            except Exception as e:
                logger.debug(f"Failed to pre-compute spans for session pool: {e}")
                # If cached version fails, get spans directly but suppress duplicate logging
                original_show_route = self.sequence_manager.sequence_manager.config.show_route
                self.sequence_manager.sequence_manager.config.show_route = False
                try:
                    spans = self.sequence_manager.sequence_manager.make_sequence(
                        mode="min_latency",
                        cache_tokens_needed=max_length
                    )
                finally:
                    self.sequence_manager.sequence_manager.config.show_route = original_show_route

            with timed_operation("session_pool_get"):
                session = self._session_pool.get_session(
                    self.sequence_manager.sequence_manager,  # Unwrap cached manager
                    max_length,
                    spans=spans  # Now spans should never be None
                )

        try:
            with session, self.use_session(session):
                yield session
        finally:
            # Return session to pool for reuse
            with timed_operation("session_pool_return"):
                self._session_pool.return_session(session)

    def __getitem__(self, ix: int | slice) -> RemoteSequential:
        return RemoteSequential(
            self.config,
            sequence_manager=self.sequence_manager[ix],
        )

    def __iter__(self):
        for block_index in range(len(self)):
            yield self[block_index]

    def __len__(self):
        return len(self.sequence_manager)

    def extra_repr(self) -> str:
        return f"modules={self.sequence_manager.block_uids[0]}..{self.sequence_manager.block_uids[-1]}"
