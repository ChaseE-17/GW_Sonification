"""gwsonify: hear gravitational waves from GWOSC.

The package is library-first. The command-line tool (:mod:`gwsonify.cli`) and the
teaching notebooks are thin layers over the functions re-exported here.

Quick start::

    import gwsonify
    result = gwsonify.sonify_data("GW150914")      # whitened detector strain
    result = gwsonify.sonify_model("GW150914")     # posterior waveform model

Most users only need :func:`sonify_data`, :func:`sonify_model` and
:func:`get_event`. The building blocks live in :mod:`gwsonify.audio`,
:mod:`gwsonify.strain`, :mod:`gwsonify.model` and :mod:`gwsonify.catalog`.
"""

__version__ = "0.1.0"


class GwsonifyError(Exception):
    """An expected, user-facing failure (bad event name, missing data, ...).

    The CLI prints the message without a traceback and exits with status 1.
    """


class MissingDependencyError(GwsonifyError):
    """An optional dependency (``pip install gwsonify[extra]``) is not installed."""


class GwsonifyWarning(UserWarning):
    """Warnings emitted by gwsonify (for example audio clipping)."""


from gwsonify.catalog import Event, get_event, list_events
from gwsonify.pipeline import Result, sonify_data, sonify_model

__all__ = [
    "Event",
    "GwsonifyError",
    "GwsonifyWarning",
    "MissingDependencyError",
    "Result",
    "__version__",
    "get_event",
    "list_events",
    "sonify_data",
    "sonify_model",
]
