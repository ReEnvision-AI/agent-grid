"""
Performance monitoring for inference sessions to track and optimize overhead.
"""
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SessionMetrics:
    """Metrics for a single session operation."""
    operation: str
    start_time: float
    end_time: float
    success: bool
    error: str | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class PerformanceMonitor:
    """
    Monitor performance of inference sessions to identify bottlenecks.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.Lock()

    def record_operation(self, operation: str, start_time: float, end_time: float,
                        success: bool = True, error: str | None = None):
        """Record metrics for an operation."""
        metric = SessionMetrics(operation, start_time, end_time, success, error)

        with self._lock:
            self._metrics[operation].append(metric)

    def get_stats(self, operation: str | None = None) -> dict[str, dict[str, float]]:
        """Get performance statistics."""
        stats = {}

        with self._lock:
            operations = [operation] if operation else self._metrics.keys()

            for op in operations:
                if op not in self._metrics or not self._metrics[op]:
                    continue

                metrics = list(self._metrics[op])
                durations = [m.duration for m in metrics]
                successes = [m.success for m in metrics]

                if durations:
                    stats[op] = {
                        'count': len(durations),
                        'avg_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations),
                        'success_rate': sum(successes) / len(successes),
                        'total_time': sum(durations),
                    }

                    # Calculate percentiles
                    sorted_durations = sorted(durations)
                    n = len(sorted_durations)
                    if n > 0:
                        stats[op]['p50'] = sorted_durations[n // 2]
                        stats[op]['p95'] = sorted_durations[int(n * 0.95)]
                        stats[op]['p99'] = sorted_durations[int(n * 0.99)]

        return stats

    def get_recent_failures(self, operation: str | None = None, count: int = 10) -> list[SessionMetrics]:
        """Get recent failed operations."""
        failures = []

        with self._lock:
            operations = [operation] if operation else self._metrics.keys()

            for op in operations:
                if op not in self._metrics:
                    continue

                for metric in reversed(self._metrics[op]):
                    if not metric.success:
                        failures.append(metric)
                        if len(failures) >= count:
                            break

                if len(failures) >= count:
                    break

        return failures[:count]

    def log_summary(self):
        """Log a performance summary."""
        stats = self.get_stats()
        if not stats:
            return

        logger.info("=== Inference Session Performance Summary ===")
        for operation, metrics in stats.items():
            logger.info(
                f"{operation}: {metrics['count']} ops, "
                f"avg={metrics['avg_duration']:.3f}s, "
                f"p95={metrics.get('p95', 0):.3f}s, "
                f"success={metrics['success_rate']:.1%}"
            )

        # Check for performance issues
        slow_operations = [
            op for op, metrics in stats.items()
            if metrics.get('p95', 0) > 5.0  # >5s p95 is slow
        ]

        if slow_operations:
            logger.warning(f"Slow operations detected: {slow_operations}")

        failed_operations = [
            op for op, metrics in stats.items()
            if metrics['success_rate'] < 0.95  # <95% success rate
        ]

        if failed_operations:
            logger.warning(f"High failure rate operations: {failed_operations}")


# Global performance monitor
_global_monitor: PerformanceMonitor | None = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


class timed_operation:
    """Context manager to time operations."""

    def __init__(self, operation: str, monitor: PerformanceMonitor | None = None):
        self.operation = operation
        self.monitor = monitor or get_performance_monitor()
        self.start_time = None
        self.error = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        success = exc_type is None
        error = str(exc_val) if exc_val else None

        self.monitor.record_operation(
            self.operation, self.start_time, end_time, success, error
        )

        return False  # Don't suppress exceptions

