#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

"""
Session pooling and caching optimizations for inference sessions.
Reduces overhead by reusing connections and caching server discovery.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from hivemind import get_logger

from agentgrid.client.inference_session import InferenceSession
from agentgrid.client.routing.sequence_manager import RemoteSequenceManager
from agentgrid.data_structures import RemoteSpanInfo

logger = get_logger(__name__)


class SessionPool:
    """
    Pool for reusing inference sessions to reduce connection overhead.
    Maintains pools of active and idle sessions with configurable limits.
    """

    def __init__(
        self,
        max_idle_sessions: int = 10,
        max_total_sessions: int = 50,
        session_ttl: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # 1 minute
    ):
        self.max_idle_sessions = max_idle_sessions
        self.max_total_sessions = max_total_sessions
        self.session_ttl = session_ttl
        self.cleanup_interval = cleanup_interval

        # Pool storage: {(span_signature, max_length): deque[PooledSession]}
        self._idle_sessions: dict[tuple[str, int], deque] = defaultdict(deque)
        self._active_sessions: set[InferenceSession] = set()
        self._session_stats: dict[InferenceSession, float] = {}  # session -> creation_time

        self._lock = threading.RLock()
        self._shutdown = False
        self._warmup_enabled = True
        
        # Start cleanup thread after all attributes are initialized
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _span_signature(self, spans: list[RemoteSpanInfo]) -> str:
        """Create a signature for a span sequence that can be used for pooling."""
        return "|".join(f"{span.peer_id}:{span.start}:{span.end}" for span in spans)

    def get_session(
        self,
        sequence_manager: RemoteSequenceManager,
        max_length: int,
        spans: list[RemoteSpanInfo] | None = None
    ) -> InferenceSession:
        """
        Get a session from pool or create new one if needed.
        
        :param sequence_manager: The sequence manager for routing
        :param max_length: Maximum sequence length for the session
        :param spans: Optional pre-computed spans to use (avoids recomputation)
        :returns: Ready-to-use InferenceSession
        """
        if spans is None:
            # Fall back to creating spans, but this adds overhead  
            spans = sequence_manager.make_sequence(mode="min_latency", cache_tokens_needed=max_length)

        span_sig = self._span_signature(spans)
        pool_key = (span_sig, max_length)

        with self._lock:
            # Try to get from idle pool first
            idle_queue = self._idle_sessions.get(pool_key)
            if idle_queue:
                while idle_queue:
                    pooled_session = idle_queue.popleft()
                    if not pooled_session.session._closed and self._is_session_valid(pooled_session):
                        # Reset session state for reuse
                        pooled_session.session.position = 0
                        self._active_sessions.add(pooled_session.session)
                        logger.debug(f"Reused session from pool: {span_sig}")
                        return pooled_session.session
                    else:
                        # Session is invalid, clean it up
                        self._cleanup_session(pooled_session.session)

            # Create new session if pool empty or sessions invalid
            if len(self._active_sessions) >= self.max_total_sessions:
                # Pool is full, clean up oldest sessions
                self._force_cleanup()

            # Create session but don't enter context yet - let caller handle that
            session = InferenceSession(sequence_manager, max_length)
            self._active_sessions.add(session)
            self._session_stats[session] = time.time()
            logger.debug(f"Created new session: {span_sig}")
            return session

    def return_session(self, session: InferenceSession):
        """
        Return a session to the pool for reuse.
        
        :param session: The session to return to pool
        """
        with self._lock:
            if session not in self._active_sessions:
                return  # Session not tracked or already returned

            self._active_sessions.remove(session)

            if session._closed or not self._is_session_healthy(session):
                # Session is not reusable
                self._cleanup_session(session)
                return

            # Determine pool key based on session's current state
            if hasattr(session, '_server_sessions') and session._server_sessions:
                spans = [s.span for s in session._server_sessions if hasattr(s, 'span')]
                if spans:
                    span_sig = self._span_signature(spans)
                    pool_key = (span_sig, session._max_length)

                    idle_queue = self._idle_sessions[pool_key]
                    if len(idle_queue) < self.max_idle_sessions:
                        pooled_session = PooledSession(session, time.time())
                        idle_queue.append(pooled_session)
                        logger.debug(f"Returned session to pool: {span_sig}")
                        return

            # If we couldn't pool it, clean it up
            self._cleanup_session(session)

    def _is_session_valid(self, pooled_session) -> bool:
        """Check if a pooled session is still valid for reuse."""
        age = time.time() - pooled_session.pooled_time
        return age < self.session_ttl and not pooled_session.session._closed

    def _is_session_healthy(self, session: InferenceSession) -> bool:
        """Check if a session is healthy enough for pooling."""
        try:
            # Basic health checks
            return (
                not session._closed and
                hasattr(session, '_server_sessions') and
                not any(getattr(s, 'closed', False) for s in session._server_sessions)
            )
        except Exception:
            return False

    def _cleanup_session(self, session: InferenceSession):
        """Clean up a session that's no longer usable."""
        try:
            if not session._closed:
                session.close()
        except Exception:
            pass
        finally:
            self._session_stats.pop(session, None)

    def _cleanup_loop(self):
        """Background cleanup of expired sessions."""
        while not self._shutdown:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_sessions()
            except Exception as e:
                logger.debug(f"Error in session pool cleanup: {e}")

    def _cleanup_expired_sessions(self):
        """Remove expired sessions from the pool."""
        current_time = time.time()

        with self._lock:
            # Clean up idle sessions
            for pool_key, idle_queue in list(self._idle_sessions.items()):
                expired_count = 0
                while idle_queue:
                    pooled_session = idle_queue[0]
                    if current_time - pooled_session.pooled_time > self.session_ttl:
                        expired_session = idle_queue.popleft()
                        self._cleanup_session(expired_session.session)
                        expired_count += 1
                    else:
                        break  # Queue is ordered by time, so we can stop here

                if not idle_queue:
                    del self._idle_sessions[pool_key]
                elif expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired sessions for {pool_key}")

    def _force_cleanup(self):
        """Force cleanup of oldest active sessions when pool is full."""
        if not self._active_sessions:
            return

        # Find oldest active session
        oldest_session = min(self._active_sessions, key=lambda s: self._session_stats.get(s, 0))
        self._active_sessions.remove(oldest_session)
        self._cleanup_session(oldest_session)
        logger.debug("Force cleaned up oldest active session")

    def shutdown(self):
        """Shutdown the session pool and cleanup all sessions."""
        self._shutdown = True

        with self._lock:
            # Cleanup all sessions
            for session in list(self._active_sessions):
                self._cleanup_session(session)
            self._active_sessions.clear()

            for idle_queue in self._idle_sessions.values():
                while idle_queue:
                    pooled_session = idle_queue.popleft()
                    self._cleanup_session(pooled_session.session)
            self._idle_sessions.clear()

    def get_stats(self) -> dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            total_idle = sum(len(queue) for queue in self._idle_sessions.values())
            return {
                "active_sessions": len(self._active_sessions),
                "idle_sessions": total_idle,
                "total_sessions": len(self._active_sessions) + total_idle,
                "pool_types": len(self._idle_sessions),
            }

    def warmup_session(
        self,
        sequence_manager: RemoteSequenceManager,
        max_length: int,
        count: int = 1
    ) -> None:
        """
        Pre-create sessions and add them to the pool for faster access.
        
        :param sequence_manager: The sequence manager for routing
        :param max_length: Maximum sequence length for the sessions
        :param count: Number of sessions to pre-create
        """
        if not self._warmup_enabled:
            return

        try:
            spans = sequence_manager.make_sequence(mode="min_latency", cache_tokens_needed=max_length)
            span_sig = self._span_signature(spans)
            pool_key = (span_sig, max_length)

            with self._lock:
                idle_queue = self._idle_sessions[pool_key]
                current_count = len(idle_queue)

                for _ in range(min(count, self.max_idle_sessions - current_count)):
                    if len(self._active_sessions) >= self.max_total_sessions:
                        break

                    try:
                        session = InferenceSession(sequence_manager, max_length)
                        # Pre-enter session to establish connections
                        session.__enter__()
                        pooled_session = PooledSession(session, time.time())
                        idle_queue.append(pooled_session)
                        logger.debug(f"Pre-warmed session for {span_sig}")
                    except Exception as e:
                        logger.debug(f"Failed to pre-warm session: {e}")
                        break
        except Exception as e:
            logger.debug(f"Failed to warmup sessions: {e}")

    def set_warmup_enabled(self, enabled: bool) -> None:
        """Enable or disable session warmup."""
        self._warmup_enabled = enabled


class PooledSession:
    """Wrapper for sessions in the pool with metadata."""

    def __init__(self, session: InferenceSession, pooled_time: float):
        self.session = session
        self.pooled_time = pooled_time


class CachedSequenceManager:
    """
    Wrapper around RemoteSequenceManager that caches expensive operations.
    """

    def __init__(self, sequence_manager: RemoteSequenceManager, cache_ttl: float = 30.0):
        self.sequence_manager = sequence_manager
        self.cache_ttl = cache_ttl

        # Cache for make_sequence results: {(start, end, mode, tokens): (spans, timestamp)}
        self._sequence_cache: dict[tuple[int, int, str, int], tuple[list[RemoteSpanInfo], float]] = {}
        self._rpc_info_cache: tuple[dict, float] | None = None
        self._lock = threading.Lock()

    def make_sequence_cached(
        self,
        start_index: int = 0,
        end_index: int | None = None,
        *,
        mode: str,
        cache_tokens_needed: int | None = None,
    ) -> list[RemoteSpanInfo]:
        """Cached version of make_sequence that avoids redundant pathfinding."""
        end_index = end_index if end_index is not None else len(self.sequence_manager)
        cache_tokens = cache_tokens_needed or 0
        cache_key = (start_index, end_index, mode, cache_tokens)

        current_time = time.time()

        with self._lock:
            # Check cache first
            if cache_key in self._sequence_cache:
                cached_spans, timestamp = self._sequence_cache[cache_key]
                if current_time - timestamp < self.cache_ttl:
                    logger.debug(f"Using cached sequence for {cache_key}")
                    return cached_spans
                else:
                    # Cache expired
                    del self._sequence_cache[cache_key]

        # Cache miss or expired, compute new sequence
        spans = self.sequence_manager.make_sequence(
            start_index, end_index, mode=mode, cache_tokens_needed=cache_tokens_needed
        )

        with self._lock:
            self._sequence_cache[cache_key] = (spans, current_time)
            # Limit cache size
            if len(self._sequence_cache) > 100:
                oldest_key = min(self._sequence_cache.keys(),
                               key=lambda k: self._sequence_cache[k][1])
                del self._sequence_cache[oldest_key]

        return spans

    @property
    def rpc_info(self):
        """Cached version of rpc_info property."""
        current_time = time.time()

        with self._lock:
            if self._rpc_info_cache is not None:
                cached_info, timestamp = self._rpc_info_cache
                if current_time - timestamp < self.cache_ttl:
                    return cached_info

        # Cache miss or expired
        rpc_info = self.sequence_manager.rpc_info

        with self._lock:
            self._rpc_info_cache = (rpc_info, current_time)

        return rpc_info

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped sequence manager."""
        return getattr(self.sequence_manager, name)


# Global session pool instance
_global_session_pool: SessionPool | None = None


def get_session_pool() -> SessionPool:
    """Get the global session pool, creating it if needed."""
    global _global_session_pool
    if _global_session_pool is None:
        _global_session_pool = SessionPool()
    return _global_session_pool


def shutdown_session_pool():
    """Shutdown the global session pool."""
    global _global_session_pool
    if _global_session_pool is not None:
        _global_session_pool.shutdown()
        _global_session_pool = None

