"""Bitcoin four-year cycle phase estimation.

Answers, with explicit uncertainty, the question "is Bitcoin pre-breakout,
mid-breakout, or is the four-year cycle no longer a thing?"

    from btc_cycle import datasources, features, ensemble

    loaded = datasources.load()
    frame = features.build(loaded.frame)
    verdict = ensemble.evaluate(frame)
    print(verdict.top_phase, verdict.confidence)

Or from the shell::

    python -m btc_cycle.report
"""

from . import config, ensemble, features, labels, synthetic

__all__ = ["config", "ensemble", "features", "labels", "synthetic"]
__version__ = "0.1.0"
