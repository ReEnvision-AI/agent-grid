"""
A pytorch memory cache that can be allocated by ConnectionHandler (on cpu) and used over multiple calls to Runtime.

For now, the only purpose of this code is to ensure that allocated memory will be deleted properly.

"""
import asyncio
import contextlib
import ctypes
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from typing import AsyncContextManager, Counter, Dict, List, Optional, Sequence

import async_timeout
import torch
from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger

from agentgrid.data_structures import Handle
from agentgrid.utils.asyncio import shield_and_wait
from agentgrid.utils.misc import get_size_in_bytes

logger = get_logger(__name__)


class MemoryCache:
    """A shared cache for storing tensors that persist across calls. Main use case: storing past attention KVs"""

    def __init__(self, max_size_bytes: Optional[int], max_alloc_timeout: Optional[float] = None):
        self.max_size_bytes = max_size_bytes if max_size_bytes is not None else (2**64 - 1)
        self.max_alloc_timeout = max_alloc_timeout
        self._lock_metadata = mp.Lock()
        self._current_size = mp.Value(ctypes.c_int64, 0, lock=False)
        self._pooled_size_bytes = mp.Value(ctypes.c_int64, 0, lock=True)
        self._enqueued_size = mp.Value(ctypes.c_int64, 0, lock=True)
        self._handle_counter = mp.Value(ctypes.c_int64, 0, lock=False)
        self._allocated_tensors: Dict[Handle, torch.Tensor] = {}
        # New free pools structure: {device: {dtype: OrderedDict(numel: [tensors])}}
        self._free_pools: Dict[torch.device, Dict[torch.dtype, "OrderedDict[int, List[torch.Tensor]]"]] = {}
        self.runtime_pid = os.getpid()

        self._pipe_recv, self._pipe_send = mp.Pipe(duplex=False)  # any ConnectionHandler -> runtime
        self._lock_acquire_memory = mp.Lock()
        self._memory_freed_event = mp.Event()

        # Compaction parameters
        self.COMPACTION_THRESHOLD = 10  # Compact if a bucket has more than this many tensors
        self._compaction_counter = 0
        self._compaction_calls_threshold = 100  # Base interval for compaction checks
        self._adaptive_compaction_min_interval = 50  # Minimum calls between compaction checks
        self._adaptive_compaction_max_interval = 500  # Maximum calls between compaction checks
        self._memory_pressure_threshold = 0.8  # Compact more aggressively when cache is 80% full
        self._last_compaction_found_work = False  # Track if last compaction actually did work

        # Memory monitoring
        self._allocation_count = 0
        self._eviction_count = 0
        self._compaction_count = 0
        self._last_monitoring_log = time.time()
        self._monitoring_interval = 60.0  # Log memory stats every 60 seconds

        # Predictive allocation
        self._session_patterns: Dict[str, List[List[TensorDescriptor]]] = {}
        self._active_sessions: Dict[str, List[TensorDescriptor]] = {}
        self._pre_allocated_tensors: Dict[str, List[torch.Tensor]] = {}
        self._pre_allocation_timestamps: Dict[str, float] = {}
        self._pre_allocation_timeout = 300.0  # 5 minutes timeout for pre-allocated tensors

    @property
    def current_size_bytes(self) -> int:
        return self._current_size.value

    @current_size_bytes.setter
    def current_size_bytes(self, value: int):
        self._current_size.value = value

    @property
    def enqueued_size_bytes(self) -> int:
        return self._enqueued_size.value

    @enqueued_size_bytes.setter
    def enqueued_size_bytes(self, value: int):
        self._enqueued_size.value = value

    @property
    def bytes_left(self) -> int:
        with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
            total_used = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
            return max(0, self.max_size_bytes - total_used)

    @property
    def handle_counter(self) -> int:
        return self._handle_counter.value

    @handle_counter.setter
    def handle_counter(self, value: int):
        self._handle_counter.value = value

    def end_session(self, session_id: str):
        """
        Signals the end of an allocation session, moving the session's allocation
        history to the stored patterns for future prediction.
        """
        assert os.getpid() != self.runtime_pid, "This method must be called from a ConnectionHandler"
        with self._lock_metadata:
            self._pipe_send.send((None, None, {"command": "end_session", "session_id": session_id}))

    @contextlib.asynccontextmanager
    async def allocate_cache(
        self, *descriptors: TensorDescriptor, timeout: float, session_id: Optional[str] = None
    ) -> AsyncContextManager[Sequence[Handle]]:
        """
        Create a handle that is associated with buffers on unique device. If cache full, raises AllocationFailed.

        :param descriptors: one or more tensors tensor of this size, dtype, etc
        :param timeout: optional maximum time to wait for cache allocation; None (default) means no time limit
        :param session_id: optional identifier for tracking allocation patterns for prediction

        :note: if descriptors reside on different devices, it is expected that they are approximately balanced across devices;
          if not, it will count maximum tensor allocation across devices for the purposes of size limit

        :note: This function should be called by connection handlers, it can be called concurrently from multiple processes.
        Furthermore, it can be called concurrently with at most one use_cache call in runtime.
        """
        assert os.getpid() != self.runtime_pid, "must be called by a ConnectionHandler, not runtime"
        assert all(descr.device is not None for descr in descriptors), "please specify allocated devices"
        if self.max_alloc_timeout is not None:
            timeout = min(timeout, self.max_alloc_timeout)
        max_alloc_size = self.get_allocation_size(*descriptors)

        gib = 1024**3
        with self._pooled_size_bytes.get_lock():
            pooled_size = self._pooled_size_bytes.value
        cur_size, max_size = self.current_size_bytes + pooled_size, self.max_size_bytes
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        logger.info(
            f"rpc_inference.wait_for_alloc(size={max_alloc_size / gib:.2f} GiB), "
            f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({cur_size / max_size * 100:.1f}%)"
        )

        alloc_task = asyncio.create_task(self._schedule_alloc(max_alloc_size, *descriptors, timeout=timeout, session_id=session_id))
        try:
            handles = await shield_and_wait(alloc_task)
            logger.info(f"rpc_inference.alloc_done(size={max_alloc_size / gib:.2f} GiB)")
            yield handles
        finally:
            self._free(max_alloc_size, alloc_task)

    @staticmethod
    def get_allocation_size(*descriptors: TensorDescriptor) -> int:
        """Return the memory size (bytes) to be allocated on a device. If there are many devices, return maximum"""
        if not descriptors:
            return 0

        alloc_size_by_device = Counter()
        for descr in descriptors:
            tensor_size = descr.numel() * get_size_in_bytes(descr.dtype)
            alloc_size_by_device[descr.device] += tensor_size
        return max(alloc_size_by_device.values())

    async def _schedule_alloc(
        self, alloc_size: int, *descriptors: TensorDescriptor, timeout: Optional[float], session_id: Optional[str] = None
    ) -> Sequence[Handle]:
        """
        This method should be called inside asyncio.shield() because:
            - hivemind.utils.enter_asynchronously() does not always release the lock on cancellation
        """
        try:
            async with self._wait_for_free_memory(alloc_size, timeout):
                with self._lock_metadata:
                    handles = tuple(int(self.handle_counter) + i for i in range(len(descriptors)))
                    self.current_size_bytes += alloc_size
                    self.handle_counter += len(handles)  # note: this will eventually overflow and it is okay
                    self._allocation_count += 1
                    command_info = {"session_id": session_id} if session_id else None
                    self._pipe_send.send((handles, descriptors, command_info))
                    self._log_memory_stats()
                    return handles
        except TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} (timeout={timeout})")

    @contextlib.asynccontextmanager
    async def _wait_for_free_memory(self, alloc_size: int, timeout: Optional[float]):
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()

        with self._enqueued_size.get_lock():
            self._enqueued_size.value += alloc_size
        allocated = False
        try:
            context_manager = async_timeout.timeout(timeout) if timeout != 0 else contextlib.AsyncExitStack()
            # contextlib.AsyncExitStack() is used as a null context here
            async with context_manager:
                with self._pooled_size_bytes.get_lock():
                    total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
                if timeout == 0 and total_size > self.max_size_bytes:
                    raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")

                async with enter_asynchronously(self._lock_acquire_memory):
                    with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
                        current_total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
                    if current_total_size + alloc_size > self.max_size_bytes:
                        if timeout == 0:
                            raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")
                        elapsed_time = time.perf_counter() - start_time
                        remaining_timeout = max(0.0, timeout - elapsed_time) if timeout is not None else None
                        await loop.run_in_executor(None, self._wait_until_available, alloc_size, remaining_timeout)

                allocated = True
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size
                yield
        except asyncio.TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} within {timeout} seconds")
        finally:
            if not allocated:
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size

    def _free(self, alloc_size: int, alloc_task: asyncio.Task):
        if alloc_task.exception() is not None:
            return
        handles = alloc_task.result()

        with self._lock_metadata:
            self._pipe_send.send((handles, None, None))  # signal runtime to free these handles
            self.current_size_bytes -= alloc_size
        self._memory_freed_event.set()

    def _wait_until_available(self, allocated_size: int, timeout: Optional[float] = None):
        # note: this function should only be called inside _lock_acquire_memory!
        if allocated_size > self.max_size_bytes:
            raise AllocationFailed(
                f"Could not allocate {allocated_size} bytes, max cache size = {self.max_size_bytes} bytes"
            )
        timeout = timeout if timeout != float("inf") else None
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
                current_total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
            if current_total_size + allocated_size <= self.max_size_bytes:
                break

            remaining_time = None if timeout is None else deadline - time.perf_counter()
            if remaining_time is not None and remaining_time <= 0:
                raise AllocationFailed(
                    f"Server's attention cache is full, failed to allocate {allocated_size} bytes in {timeout} seconds"
                )

            if not self._memory_freed_event.wait(remaining_time):
                raise AllocationFailed(
                    f"Server's attention cache is full, failed to allocate {allocated_size} bytes in {timeout} seconds"
                )
            self._memory_freed_event.clear()

    def _evict_memory(self, bytes_to_free: int):
        """Evicts tensors from free pools in LRU order until at least `bytes_to_free` are freed."""
        bytes_freed = 0
        devices = list(self._free_pools.keys())
        for device in devices:
            if bytes_freed >= bytes_to_free:
                break
            dtypes = list(self._free_pools.get(device, {}).keys())
            for dtype in dtypes:
                if bytes_freed >= bytes_to_free:
                    break
                numels = list(self._free_pools[device].get(dtype, {}).keys())
                for numel in numels:
                    if bytes_freed >= bytes_to_free:
                        break
                    pool = self._free_pools[device][dtype][numel]
                    while pool and bytes_freed < bytes_to_free:
                        tensor = pool.pop()
                        tensor_size = tensor.numel() * get_size_in_bytes(tensor.dtype)
                        with self._pooled_size_bytes.get_lock():
                            self._pooled_size_bytes.value -= tensor_size
                        bytes_freed += tensor_size
                        del tensor  # Let python GC reclaim memory
                    if not pool:
                        del self._free_pools[device][dtype][numel]

                if not self._free_pools[device][dtype]:
                    del self._free_pools[device][dtype]
            if not self._free_pools[device]:
                del self._free_pools[device]

        if bytes_freed > 0:
            self._memory_freed_event.set()
            self._eviction_count += 1
        logger.debug(f"Evicted {bytes_freed / 1024**2:.2f} MB from memory cache pools.")

    def _needs_compaction(self) -> bool:
        """Check if any pools have enough tensors to warrant compaction."""
        for device_pools in self._free_pools.values():
            for dtype_pools in device_pools.values():
                for pool in dtype_pools.values():
                    if len(pool) >= self.COMPACTION_THRESHOLD:
                        return True
        return False

    def _calculate_fragmentation_score(self) -> float:
        """Calculate a fragmentation score based on pool distribution."""
        total_tensors = 0
        total_pools = 0
        
        for device_pools in self._free_pools.values():
            for dtype_pools in device_pools.values():
                for pool in dtype_pools.values():
                    if pool:  # Only count non-empty pools
                        total_tensors += len(pool)
                        total_pools += 1
        
        if total_pools == 0:
            return 0.0
        
        # Higher score means more fragmentation (many small pools)
        return total_pools / max(1, total_tensors)

    def _get_memory_pressure(self) -> float:
        """Get current memory pressure as a ratio of used/max memory."""
        with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
            total_used = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
        return total_used / self.max_size_bytes if self.max_size_bytes != 2**64 - 1 else 0.0

    def _calculate_adaptive_interval(self) -> int:
        """Calculate adaptive compaction interval based on memory pressure and fragmentation."""
        memory_pressure = self._get_memory_pressure()
        fragmentation_score = self._calculate_fragmentation_score()
        
        # Base interval adjustment factors
        pressure_factor = 1.0
        fragmentation_factor = 1.0
        recency_factor = 1.0
        
        # Adjust based on memory pressure - compact more often when under pressure
        if memory_pressure > self._memory_pressure_threshold:
            pressure_factor = 0.5  # Halve the interval (compact twice as often)
        elif memory_pressure > 0.6:
            pressure_factor = 0.75  # Compact 33% more often
        
        # Adjust based on fragmentation - more fragmented pools need more frequent compaction
        if fragmentation_score > 0.8:
            fragmentation_factor = 0.6  # Compact more often with high fragmentation
        elif fragmentation_score > 0.5:
            fragmentation_factor = 0.8
        
        # Adjust based on whether last compaction found work
        if not self._last_compaction_found_work:
            recency_factor = 1.5  # Increase interval if last compaction didn't find work
        
        # Calculate final interval
        adaptive_interval = int(
            self._compaction_calls_threshold * pressure_factor * fragmentation_factor * recency_factor
        )
        
        # Clamp to min/max bounds
        return max(
            self._adaptive_compaction_min_interval,
            min(self._adaptive_compaction_max_interval, adaptive_interval)
        )

    def _compact_memory_pools(self):
        """
        An adaptive compaction strategy that runs when pools actually need compaction.
        It finds buckets with many tensors and attempts to merge them into larger tensors.
        """
        if not self._needs_compaction():
            logger.debug("Skipping compaction - no pools need compaction")
            self._last_compaction_found_work = False
            return
        
        logger.debug("Running memory pool compaction...")
        compaction_work_done = False
        
        for device, device_pools in self._free_pools.items():
            for dtype, dtype_pools in device_pools.items():
                # Find the bucket with the most tensors (candidate for compaction)
                # We iterate over a copy of keys since we might modify the dictionary
                for numel, pool in list(dtype_pools.items()):
                    if len(pool) >= self.COMPACTION_THRESHOLD:
                        logger.debug(
                            f"Compacting pool on {device}:{dtype} with {len(pool)} tensors of size {numel}"
                        )
                        initial_pool_size = len(pool)
                        
                        while len(pool) >= 2:
                            t1 = pool.pop()
                            t2 = pool.pop()
                            new_numel = t1.numel() + t2.numel()

                            try:
                                # The accounting for _pooled_size_bytes does not change here,
                                # as we are replacing two tensors with one of their combined size.
                                new_tensor = torch.empty(new_numel, dtype=dtype, device=device)
                            except Exception as e:
                                logger.warning(f"Compaction failed to allocate new tensor: {e}")
                                # If allocation fails, put the tensors back and stop.
                                pool.append(t1)
                                pool.append(t2)
                                break

                            # Add the new, larger tensor to its corresponding pool
                            new_pool = dtype_pools.setdefault(new_numel, [])
                            new_pool.append(new_tensor)
                            dtype_pools.move_to_end(new_numel)
                            compaction_work_done = True
                        
                        logger.debug(f"Compaction finished for pool. Size: {initial_pool_size} -> {len(pool)}")
        
        self._last_compaction_found_work = compaction_work_done
        if compaction_work_done:
            logger.debug("Memory pool compaction completed with work done")
        else:
            logger.debug("Memory pool compaction completed with no work needed")

    def _predict_and_preallocate(self, session_id: str):
        """Compares active session with past patterns and pre-allocates the next tensor if a match is found."""
        if session_id not in self._active_sessions or session_id in self._pre_allocated_tensors:
            return  # No history to predict from or already pre-allocated

        history = self._active_sessions[session_id]
        if not history:
            return

        # Clean up expired pre-allocations
        self._cleanup_expired_preallocations()

        for pattern in self._session_patterns.get(session_id, []):
            if len(pattern) > len(history) and pattern[: len(history)] == history:
                # Found a matching pattern, predict the next step
                next_descriptor = pattern[len(history)]
                logger.debug(f"Predictive allocation for session {session_id}: found pattern, next is {next_descriptor}")

                # Attempt to find and reserve a tensor from free pools
                numel = next_descriptor.numel()
                device_pools = self._free_pools.get(next_descriptor.device)
                if device_pools:
                    dtype_pools = device_pools.get(next_descriptor.dtype)
                    if dtype_pools and numel in dtype_pools and dtype_pools[numel]:
                        pre_allocated_tensor = dtype_pools[numel].pop()
                        self._pre_allocated_tensors.setdefault(session_id, []).append(pre_allocated_tensor)
                        self._pre_allocation_timestamps[session_id] = time.time()
                        logger.debug(f"Pre-allocated tensor for session {session_id}")
                        # We found a prediction and pre-allocated, we are done for this step
                        return

    @contextlib.contextmanager
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        assert os.getpid() == self.runtime_pid

        # Step 1: Adaptive periodic compaction
        self._compaction_counter += 1
        adaptive_interval = self._calculate_adaptive_interval()
        
        if self._compaction_counter >= adaptive_interval:
            self._compact_memory_pools()
            self._compaction_counter = 0
            self._compaction_count += 1

        # Step 2: Evict memory if cache is over budget
        with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
            total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
        if total_size > self.max_size_bytes:
            self._evict_memory(total_size - self.max_size_bytes)

        # Step 3: read creation/deletion requests from connection handlers
        while self._pipe_recv.poll():
            message = self._pipe_recv.recv()
            recv_handles, recv_data, command_info = message if len(message) == 3 else (message[0], message[1], None)

            if command_info and command_info.get("command") == "end_session":
                session_id = command_info["session_id"]
                if session_id in self._active_sessions:
                    history = self._active_sessions.pop(session_id)
                    patterns = self._session_patterns.setdefault(session_id, [])
                    patterns.append(history)
                    logger.debug(f"Ended session {session_id}, stored allocation pattern with {len(history)} steps.")
                if session_id in self._pre_allocated_tensors: # Clean up any leftover pre-allocations
                    for tensor in self._pre_allocated_tensors.pop(session_id):
                        # Return tensor to the free pool
                        descr = TensorDescriptor.from_tensor(tensor)
                        device_pool = self._free_pools.setdefault(descr.device, {})
                        dtype_pool = device_pool.setdefault(descr.dtype, OrderedDict())
                        numel_pool = dtype_pool.setdefault(tensor.numel(), [])
                        numel_pool.append(tensor)
                    self._pre_allocation_timestamps.pop(session_id, None)

            if recv_data is not None:  # create new tensors
                assert len(recv_handles) == len(recv_data)
                session_id = command_info.get("session_id") if command_info else None
                if session_id:
                    session_history = self._active_sessions.setdefault(session_id, [])
                    session_history.extend(recv_data)

                for handle, descr in zip(recv_handles, recv_data):
                    reused_tensor = None
                    numel = descr.numel()

                    # Step 3a: Try to fulfill from pre-allocated tensors first
                    if session_id and session_id in self._pre_allocated_tensors:
                        pre_allocated_pool = self._pre_allocated_tensors[session_id]
                        for i, tensor in enumerate(pre_allocated_pool):
                            if tensor.numel() == numel and tensor.dtype == descr.dtype and tensor.device == descr.device:
                                reused_tensor = pre_allocated_pool.pop(i)
                                logger.debug(f"Used pre-allocated tensor for session {session_id}")
                                break
                        if not pre_allocated_pool:
                            del self._pre_allocated_tensors[session_id]
                            self._pre_allocation_timestamps.pop(session_id, None)

                    # Step 3b: If not found, try to fulfill from general free pools
                    if reused_tensor is None:
                        device_pools = self._free_pools.get(descr.device)
                        if device_pools:
                            dtype_pools = device_pools.get(descr.dtype)
                            if dtype_pools and numel in dtype_pools and dtype_pools[numel]:
                                reused_tensor = dtype_pools[numel].pop()
                                dtype_pools.move_to_end(numel)

                    if reused_tensor is not None:
                        self._allocated_tensors[handle] = reused_tensor.view(descr.shape)
                        tensor_size = reused_tensor.numel() * get_size_in_bytes(reused_tensor.dtype)
                        with self._pooled_size_bytes.get_lock():
                            self._pooled_size_bytes.value -= tensor_size
                    else:
                        self._allocated_tensors[handle] = torch.empty(
                            descr.shape, dtype=descr.dtype, device=descr.device
                        )
                    assert handle in self._allocated_tensors, f"Sanity check failed: no such handle ({handle})"

                if session_id: # After fulfilling request, try to predict and pre-allocate the next one
                    self._predict_and_preallocate(session_id)

            elif recv_handles is not None:  # delete tensors by handle
                for handle in recv_handles:
                    if handle not in self._allocated_tensors:
                        logger.warning(
                            f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                        )
                        continue
                    tensor = self._allocated_tensors.pop(handle)
                    descr = TensorDescriptor.from_tensor(tensor)
                    numel = tensor.numel()

                    device_pool = self._free_pools.setdefault(descr.device, {})
                    dtype_pool = device_pool.setdefault(descr.dtype, OrderedDict())
                    numel_pool = dtype_pool.setdefault(numel, [])
                    numel_pool.append(tensor)
                    dtype_pool.move_to_end(numel)  # Mark as recently used

                    tensor_size = numel * get_size_in_bytes(descr.dtype)
                    with self._pooled_size_bytes.get_lock():
                        self._pooled_size_bytes.value += tensor_size

        # Step 4: Yield tensors
        yield tuple(self._allocated_tensors[handle] for handle in handles)

    def recycle_tensors(self, tensors: Sequence[torch.Tensor]) -> None:
        """Return temporary tensors back to the free pool for future reuse."""
        assert os.getpid() == self.runtime_pid
        for tensor in tensors:
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type != "mps":
                continue
            descr = TensorDescriptor.from_tensor(tensor)
            numel = tensor.numel()
            device_pool = self._free_pools.setdefault(descr.device, {})
            dtype_pool = device_pool.setdefault(descr.dtype, OrderedDict())
            numel_pool = dtype_pool.setdefault(numel, [])
            numel_pool.append(tensor)
            dtype_pool.move_to_end(numel)
            tensor_size = numel * get_size_in_bytes(descr.dtype)
            with self._pooled_size_bytes.get_lock():
                self._pooled_size_bytes.value += tensor_size

    def _cleanup_expired_preallocations(self):
        """Clean up pre-allocated tensors that have expired due to timeout."""
        current_time = time.time()
        expired_sessions = []

        for session_id, timestamp in self._pre_allocation_timestamps.items():
            if current_time - timestamp > self._pre_allocation_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            logger.warning(f"Cleaning up expired pre-allocations for session {session_id}")
            if session_id in self._pre_allocated_tensors:
                for tensor in self._pre_allocated_tensors.pop(session_id):
                    # Return tensor to the free pool
                    descr = TensorDescriptor.from_tensor(tensor)
                    device_pool = self._free_pools.setdefault(descr.device, {})
                    dtype_pool = device_pool.setdefault(descr.dtype, OrderedDict())
                    numel_pool = dtype_pool.setdefault(tensor.numel(), [])
                    numel_pool.append(tensor)
                self._pre_allocation_timestamps.pop(session_id, None)
                continue

    def _log_memory_stats(self):
        """Log memory statistics for monitoring purposes."""
        current_time = time.time()
        if current_time - self._last_monitoring_log >= self._monitoring_interval:
            gib = 1024**3
            with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
                total_used = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
                memory_pressure = total_used / self.max_size_bytes if self.max_size_bytes != 2**64 - 1 else 0.0

            logger.info(
                f"Memory Cache Stats - "
                f"Used: {total_used / gib:.2f} GiB ({memory_pressure * 100:.1f}%), "
                f"Available: {self.bytes_left / gib:.2f} GiB, "
                f"Allocations: {self._allocation_count}, "
                f"Evictions: {self._eviction_count}, "
                f"Compactions: {self._compaction_count}, "
                f"Active Sessions: {len(self._active_sessions)}, "
                f"Pre-allocated: {len(self._pre_allocated_tensors)}"
            )

            # Log warning if memory pressure is high
            if memory_pressure > 0.9:
                logger.warning(f"High memory pressure detected: {memory_pressure * 100:.1f}%")
            elif memory_pressure > 0.75:
                logger.info(f"Elevated memory pressure: {memory_pressure * 100:.1f}%")

            self._last_monitoring_log = current_time


class AllocationFailed(Exception):
    pass
