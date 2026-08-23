"""Process-wide runtime state: graceful shutdown handling and logging setup."""
import logging
import signal
from threading import Event


class GracefulShutdown:
    """Thread-safe flag that lets long-running loops stop cleanly on Ctrl+C.

    Every worker loop in the pipeline (discovery, extraction, derivation)
    checks `SHUTDOWN.requested` between units of work so that a SIGINT/SIGTERM
    results in the current checkpoint being flushed instead of a hard kill.
    """

    def __init__(self):
        self._shutdown_event = Event()

    def request_shutdown(self):
        self._shutdown_event.set()

    @property
    def requested(self) -> bool:
        return self._shutdown_event.is_set()


SHUTDOWN = GracefulShutdown()


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Attach a single stream handler to the shared 'secodash' logger.

    Safe to call multiple times: handlers are only added once, so repeated
    calls just adjust the log level.
    """
    logger = logging.getLogger("secodash")
    if not logger.handlers:
        level = logging.DEBUG if verbose else logging.INFO
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)
    return logger


logger = logging.getLogger("secodash")


def install_signal_handlers():
    """Route SIGINT/SIGTERM to GracefulShutdown instead of killing the process outright."""
    def _handler(signum, frame):
        logger.warning("Interrupt received. Triggering graceful shutdown (checkpoints will be saved)...")
        SHUTDOWN.request_shutdown()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
