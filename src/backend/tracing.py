"""Local execution metadata; CV text and credentials are never logged here."""

import logging
from contextvars import ContextVar
from time import perf_counter

from langchain_core.runnables import RunnableConfig

thread_id = ContextVar("search_thread_id", default=None)
logger = logging.getLogger("jd_matching.trace")


def traced(node):
    def run(state: dict, config: RunnableConfig):
        token = thread_id.set(config.get("configurable", {}).get("thread_id"))
        started = perf_counter()
        try:
            return node(state)
        finally:
            logger.info("node=%s search_id=%s cv_id=%s elapsed_ms=%.3f",
                        node.__name__, thread_id.get(), state.get("cv_id"),
                        (perf_counter() - started) * 1000)
            thread_id.reset(token)
    run.__name__ = node.__name__
    return run
