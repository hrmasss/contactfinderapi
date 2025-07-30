import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore


logger = logging.getLogger(__name__)

# Global registry for scheduler callbacks
_callback_registry = {}


async def _execute_scheduled_callback(callback_name: str, *args, **kwargs):
    """Standalone function to execute scheduled callbacks"""
    try:
        if callback_name not in _callback_registry:
            logger.error(f"Callback '{callback_name}' not found in registry")
            return

        callback = _callback_registry[callback_name]
        if hasattr(callback, "__call__"):
            result = callback(*args, **kwargs)
            # Handle async callbacks
            if hasattr(result, "__await__"):
                await result
    except Exception as e:
        logger.error(f"Callback '{callback_name}' failed: {e}")


class TaskScheduler:
    """Generic task scheduler with persistence"""

    def __init__(self):
        self.scheduler = None
        self.callbacks: Dict[str, Callable] = {}
        self._setup_scheduler()

    def _setup_scheduler(self):
        """Setup APScheduler with SQLite persistence"""
        jobstore = SQLAlchemyJobStore(url="sqlite:///jobs.sqlite3")
        self.scheduler = AsyncIOScheduler(
            jobstores={"default": jobstore},
            job_defaults={
                "coalesce": False,
                "max_instances": 1,
                "misfire_grace_time": 300,
            },
        )

    async def start(self):
        """Start the scheduler"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Task scheduler started")

    async def shutdown(self):
        """Graceful shutdown"""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Task scheduler shutdown complete")

    def register_callback(self, name: str, callback: Callable):
        """Register a callback function"""
        self.callbacks[name] = callback
        _callback_registry[name] = callback

    def schedule_task(
        self, callback_name: str, run_at: datetime, task_id: str, *args, **kwargs
    ) -> str:
        """Schedule a generic task"""
        if callback_name not in self.callbacks:
            raise ValueError(f"Callback '{callback_name}' not registered")

        job = self.scheduler.add_job(
            _execute_scheduled_callback,
            "date",
            run_date=run_at,
            args=[callback_name] + list(args),
            kwargs=kwargs,
            id=task_id,
        )

        return job.id

    def schedule_delayed_task(
        self, callback_name: str, delay_minutes: int, task_id: str, *args, **kwargs
    ) -> str:
        """Schedule a task with delay"""
        run_at = datetime.now() + timedelta(minutes=delay_minutes)
        return self.schedule_task(callback_name, run_at, task_id, *args, **kwargs)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        try:
            self.scheduler.remove_job(task_id)
            return True
        except Exception:
            return False


# Global scheduler instance
_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get the global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized")
    return _scheduler


def init_scheduler() -> TaskScheduler:
    """Initialize the global scheduler"""
    global _scheduler
    _scheduler = TaskScheduler()
    return _scheduler


async def start_scheduler():
    """Start the scheduler"""
    scheduler = get_scheduler()
    await scheduler.start()


async def shutdown_scheduler():
    """Shutdown the scheduler"""
    if _scheduler:
        await _scheduler.shutdown()
