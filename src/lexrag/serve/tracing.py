"""Optional Langfuse tracing (enabled when LANGFUSE_PUBLIC_KEY is set), else a no-op.

Every request is also returned with its own timings, so latency is visible
without Langfuse. Written against the Langfuse v2 Python SDK.
"""

from __future__ import annotations

import os
from contextlib import contextmanager


class _NoopSpan:
    def end(self, **kw):
        pass

    def update(self, **kw):
        pass


class Tracer:
    def __init__(self):
        self.lf = None
        if os.getenv("LANGFUSE_PUBLIC_KEY"):
            try:
                from langfuse import Langfuse

                self.lf = Langfuse()  # reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST
            except Exception as e:  # tracing must never break answering
                print(f"[tracing] disabled: {e}")

    def trace(self, name: str, **kw):
        return self.lf.trace(name=name, **kw) if self.lf else None

    @contextmanager
    def span(self, trace, name: str, **kw):
        s = trace.span(name=name, **kw) if trace is not None else _NoopSpan()
        try:
            yield s
        finally:
            s.end()

    def generation(self, trace, **kw):
        if trace is not None:
            trace.generation(**kw)

    def flush(self):
        if self.lf:
            self.lf.flush()
