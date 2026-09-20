"""CLI: ``python -m btc_cycle.report``.

    python -m btc_cycle.report                    # fetch, analyse, print
    python -m btc_cycle.report --csv my_data.csv  # offline, your own export
    python -m btc_cycle.report --json out.json    # machine-readable
    python -m btc_cycle.report --backtest         # forward returns per phase
    python -m btc_cycle.report --self-test        # synthetic sanity check
    python -m btc_cycle.report --position         # exposure + short hurdle
    python -m btc_cycle.report --strategy         # vs buy-and-hold, after costs
    python -m btc_cycle.report --hurdle-only      # short arithmetic, no data needed
    python -m btc_cycle.report --pattern-noise    # patterns in pure noise, no data needed
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config, datasources, ensemble, features, patterns, position


def _fmt_pct(value: float) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value * 100:5.1f}%"


def _print_verdict(verdict, frame: pd.DataFrame) -> None:
    line = "=" * 68
    print(line)
    print(f"BTC CYCLE PHASE READ   as of {verdict.as_of}")
    print(line)

    latest = frame.iloc[-1]
    print(f"price               {latest['price']:,.0f}")
    print(f"drawdown from ATH   {_fmt_pct(latest.get('drawdown'))}")
    print(f"Mayer multiple      {latest.get('mayer', float('nan')):.2f}")
    if np.isfinite(latest.get("ma200w_mult", np.nan)):
        print(f"200-week MA mult    {latest['ma200w_mult']:.2f}")
    if "mvrv" in frame.columns and np.isfinite(latest.get("mvrv", np.nan)):
        print(f"MVRV                {latest['mvrv']:.2f}")
    print(f"days since halving  {int(latest.get('days_since_halving', 0))}")

    print("\nPHASE POSTERIOR")
    for phase in config.PHASES:
        probability = verdict.probabilities.get(phase, float("nan"))
        bar = "#" * int(round((probability if np.isfinite(probability) else 0) * 40))
        print(f"  {config.PHASE_ZH[phase]:<10} {phase:<14} {_fmt_pct(probability)}  {bar}")

    print(f"\ntop phase           {verdict.top_phase} ({config.PHASE_ZH[verdict.top_phase]})")
    print(f"model agreement     {_fmt_pct(verdict.agreement)}")
    print(f"confidence          {_fmt_pct(verdict.confidence)}")
    print(f"cycle informative   {verdict.cycle_informative}")
    print(f"novelty flag        {verdict.novelty_flag}")

    cycle = verdict.components.get("cycle_test", {})
    if cycle:
        spectral = cycle.get("spectral", {})
        predictive = cycle.get("predictive", {})
        print("\nDOES THE CYCLE STILL EXIST?")
        if "p_value" in spectral:
            print(
                f"  spectral peak {spectral['peak_period_years']:.2f}y, "
                f"surrogate p = {spectral['p_value']:.3f} -> {spectral['verdict']}"
            )
        if "delta_r2" in predictive:
            print(
                f"  halving harmonics OOS dR2 = {predictive['delta_r2']:+.4f}, "
                f"sign-test p = {predictive['sign_test_p']:.3f} -> {predictive['verdict']}"
            )

    supervised = verdict.components.get("supervised", {}).get("validation", {})
    if "mean_accuracy" in supervised:
        print("\nHONEST VALIDATION (leave-one-cycle-out)")
        for fold in supervised["folds"]:
            print(
                f"  hold out cycle {fold['held_out_cycle']}: "
                f"acc {_fmt_pct(fold['accuracy'])} vs majority "
                f"{_fmt_pct(fold['majority_class_rate'])} "
                f"(lift {fold['lift_over_majority']:+.3f})"
            )
        print(f"  effective sample size: {supervised['effective_sample_size']} cycles")

    analog = verdict.components.get("analog", {})
    if analog.get("summary"):
        summary = analog["summary"]
        clock = analog.get("halving_clock", {})
        print("\nANALOG CONTEXT (no vote)")
        print(
            f"  position in prior cycles: {summary['position_weighted']:.2f} "
            f"(range {summary['position_min']:.2f}-{summary['position_max']:.2f})"
        )
        print(
            f"  implied days to top: {summary['days_to_top_weighted']:.0f} "
            f"(range {summary['days_to_top_min']:.0f}-{summary['days_to_top_max']:.0f})"
        )
        if clock:
            print(
                f"  days since halving {clock['current_days_since_halving']} vs "
                f"prior mean {clock['mean_days']:.0f} +/- {clock['std_days']:.0f}"
            )

    if verdict.notes:
        print("\nNOTES")
        for note in verdict.notes:
            print(f"  - {note}")
    print(line)


def _self_test(fast: bool = True) -> int:
    """Run the pipeline on synthetic data with and without a planted cycle."""
    from . import synthetic

    from .models import cycle_test

    print("Synthetic check: a series WITH a cycle, then one WITHOUT.\n")
    ok = True
    acyclic_frame = None
    for name, maker in (("cyclical", synthetic.make_cyclical),
                        ("acyclic (random walk)", synthetic.make_acyclic)):
        raw = maker()
        frame = features.build(raw)
        if name.startswith("acyclic"):
            acyclic_frame = frame

        spectral = cycle_test.spectral_test(frame["price"], n_surrogates=100 if fast else 500)
        detected = spectral.get("p_value", 1.0) <= 0.10
        expected = name == "cyclical"
        status = "PASS" if detected == expected else "FAIL"
        if detected != expected:
            ok = False
        print(
            f"  {name:<24} peak {spectral.get('peak_period_years', float('nan')):.2f}y  "
            f"p={spectral.get('p_value', float('nan')):.3f}  "
            f"detected={detected}  expected={expected}  [{status}]"
        )
    print(
        "\nIf the acyclic row reports a detected cycle, the null is not doing "
        "its job and nothing else in this package should be trusted."
    )

    if acyclic_frame is not None:
        power = cycle_test.spectral_power_analysis(
            acyclic_frame["price"],
            n_trials=25 if fast else 60,
            n_surrogates=60 if fast else 120,
        )
        if "power_by_amplitude" in power:
            print(
                f"\nHOW BIG WOULD A CYCLE HAVE TO BE TO SHOW UP IN "
                f"{power['n_cycles_observed']:.1f} CYCLES OF DATA?"
            )
            print("  log amp   peak/trough    detection rate (= power)")
            for row in power["power_by_amplitude"]:
                print(
                    f"  {row['log_amplitude']:>6.2f}   {row['peak_to_trough_multiple']:>9.1f}x"
                    f"   {row['detection_rate']:>6.0%}"
                )
            print(
                "\n  Power below ~80% means a negative result is uninformative: "
                "the cycle\n  could be real and this much history simply cannot "
                "resolve it. That is\n  the single most important number when "
                "answering 'is the cycle dead?'."
            )
    return 0 if ok else 1


def _print_hurdle(drift: float = 0.40, vol: float = 0.50) -> None:
    """Short-hurdle arithmetic. Needs no market data at all."""
    summary = position.short_case_summary(drift_annual=drift, vol_annual=vol)
    hurdle, bears = summary["hurdle"], summary["bear_legs"]

    print("=" * 68)
    print("SHORT HURDLE  (arithmetic only - no market data used)")
    print("=" * 68)

    drifts = position.historical_drift_estimates()
    print("Annualised drift between real cycle anchors:")
    for row in drifts["spans"]:
        print(f"  {row['span']:<26} {row['kind']:<24} "
              f"{row['years']:>5.2f}y  {row['annual_log_drift']:>+8.1%}/yr")
    print(f"  low-to-low median {drifts['low_to_low_median']:+.1%}/yr "
          f"(range {drifts['low_to_low_min']:+.1%} to {drifts['low_to_low_max']:+.1%})")

    print(f"\nAssumed: drift {hurdle['assumed_drift']:+.0%}/yr, "
          f"vol {hurdle['assumed_vol']:.0%}/yr, funding "
          f"{hurdle['funding_annual']:.0%}/yr, costs {hurdle['annual_cost']:.0%}/yr")
    print(f"Break-even conditional return for a short: "
          f"{hurdle['breakeven_conditional_return']:+.0%}/yr")
    print(f"Shift needed from the unconditional drift: "
          f"{hurdle['required_shift_to_breakeven']:+.0%} percentage points\n")

    print("  eff_n   std err   required point estimate   (= SE below drift)")
    for row in hurdle["by_effective_n"]:
        print(f"  {row['effective_n']:>5}   {row['standard_error']:>6.0%}   "
              f"{row['required_point_estimate']:>+18.0%}/yr   "
              f"{row['shift_in_standard_errors']:>15.1f}")

    print("\nWhat the hindsight bear legs actually delivered:")
    for leg in bears["legs"]:
        print(f"  {leg['leg']:<26} {leg['years']:>4.2f}y  "
              f"drawdown {leg['drawdown']:>+7.1%}  "
              f"annualised {leg['annual_log_return']:>+8.1%}/yr")
    print(f"  median {bears['median_annual_log_return']:+.1%}/yr")

    print(f"\n  threshold to justify a short (eff_n=8): "
          f"{summary['threshold_at_effective_n_8']:+.0%}/yr")
    print(f"  median realised bear leg:               "
          f"{summary['median_realised_bear_return']:+.0%}/yr")
    print(f"  headroom:                               {summary['headroom']:+.0%} pp")
    print(f"\n{summary['conclusion']}")
    print(f"\nCaveat on those legs: {bears['caveat']}")
    print("=" * 68)


def _print_position(frame: pd.DataFrame, verdict) -> None:
    print("\nPOSITION")
    try:
        advice = position.recommend(frame, verdict)
    except Exception as exc:
        print(f"  position read unavailable: {exc}")
        return

    print(f"  action              {advice.action}")
    print(f"  target exposure     {advice.target_exposure:+.2f}x "
          f"(1.00 = fully long spot)")
    print(f"  expected return     {advice.expected_annual_return:+.1%}/yr "
          f"(posterior-weighted)")
    if advice.refused_short:
        print("  short               REFUSED by the hurdle gate")

    if advice.phase_stats:
        print("\n  out-of-sample forward returns by phase (annualised):")
        for phase, stats in advice.phase_stats.items():
            se = stats.get("bootstrap_se", float("nan"))
            print(f"    {config.PHASE_ZH.get(phase, phase):<10} "
                  f"{stats['annualised_mean']:>+7.1%}  +/- {se:>5.1%}  "
                  f"eff_n {stats['effective_n']:>5}")

    if advice.rationale:
        print("\n  reasoning:")
        for line in advice.rationale:
            print(f"    - {line}")


def _pattern_row(name: str, profile: dict) -> str:
    pivot = profile.get("from_pivot", {})
    signal = profile.get("from_signal", {})
    return (
        f"  {name:<20} {profile['matches_per_series']:>6.1f} "
        f"{profile['median_lag_bars']:>6.0f}   "
        f"{pivot.get('mean', float('nan')):>+7.2%} "
        f"({pivot.get('standard_errors', float('nan')):>5.1f} se)   "
        f"{signal.get('mean', float('nan')):>+7.2%} "
        f"({signal.get('standard_errors', float('nan')):>5.1f} se)"
    )


def _print_pattern_noise(n_series: int = 100) -> None:
    """What chart patterns do in data known to contain nothing."""
    rng = np.random.default_rng(0)
    walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0009, 0.035, 4000)))

    print("=" * 78)
    print("CHART PATTERNS IN PURE NOISE  (random walk, nothing to predict)")
    print("=" * 78)
    print(f"  {'pattern':<20} {'per':>6} {'lag':>6}   {'measured from pivot':>21}"
          f"   {'measured from signal':>22}")
    print(f"  {'':<20} {'series':>6} {'bars':>6}")

    baseline = None
    for name in patterns.DETECTORS:
        profile = patterns.noise_profile(walk, name, n_series=n_series)
        baseline = profile["baseline"]
        print(_pattern_row(name, profile))

    if baseline:
        print(f"\n  unconditional baseline: {baseline['mean']:+.2%}")
    print(
        "\n  Every pattern is hugely 'significant' from its own pivot and "
        "worth nothing\n  from the bar you could have acted on - in data with "
        "no signal in it at all.\n  The first column is the pattern's "
        "definition looking forward, not a forecast."
    )
    print("=" * 78)


def _print_pattern_audit(prices, n_surrogates: int = 100) -> None:
    print("\nCHART PATTERN AUDIT (this series)")
    print(f"  {'pattern':<20} {'found':>6}  {'from pivot':>16}  {'from signal':>16}"
          f"  {'in noise':>9}")
    for name in patterns.DETECTORS:
        try:
            result = patterns.edge_test(prices, name, n_surrogates=n_surrogates)
        except Exception as exc:
            print(f"  {name:<20} failed: {exc}")
            continue
        audit = result["lookahead_audit"]
        if audit.get("n_matches", 0) == 0:
            print(f"  {name:<20} {'0':>6}  (absent)")
            continue
        pivot = audit["from_pivot"]
        signal = audit["from_signal"]
        null = result["null_frequency"]
        print(
            f"  {name:<20} {audit['n_matches']:>6}  "
            f"{pivot.get('standard_errors', float('nan')):>13.1f} se  "
            f"{signal.get('standard_errors', float('nan')):>13.1f} se  "
            f"{null['series_with_at_least_one']:>8.0%}"
        )
        if result["hindsight_only"]:
            print(f"  {'':<20} -> hindsight only; not tradeable")
    print(
        "\n  'in noise' = share of matched random walks containing the pattern. "
        "Near 100%\n  means its presence alone carries no information. Run "
        "--pattern-noise for the\n  reference distribution."
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bitcoin four-year cycle phase read")
    parser.add_argument("--csv", help="offline CSV export (date,price[,mvrv,realized_price,...])")
    parser.add_argument("--json", help="write the full result to this path")
    parser.add_argument("--fast", action="store_true", help="fewer surrogates; quicker, noisier")
    parser.add_argument("--backtest", action="store_true", help="forward returns per phase call")
    parser.add_argument("--position", action="store_true", help="position sizing and the short hurdle")
    parser.add_argument("--strategy", action="store_true", help="walk-forward strategy vs buy-and-hold")
    parser.add_argument("--allow-short", action="store_true", help="let the backtest take shorts")
    parser.add_argument("--hurdle-only", action="store_true", help="short-hurdle arithmetic, no market data needed")
    parser.add_argument("--pattern-audit", action="store_true", help="test chart patterns for look-ahead bias")
    parser.add_argument("--pattern-noise", action="store_true", help="what patterns look like in pure noise; no data needed")
    parser.add_argument("--no-supervised", action="store_true")
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="synthetic sanity check, no network")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test(fast=args.fast)

    if args.hurdle_only:
        _print_hurdle()
        return 0

    if args.pattern_noise:
        _print_pattern_noise()
        return 0

    try:
        loaded = datasources.load(
            csv_path=args.csv, with_macro=not args.no_macro, use_cache=not args.no_cache
        )
    except datasources.DataUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for warning in loaded.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"source: {loaded.source}  rows: {len(loaded.frame)}  "
          f"on-chain: {loaded.has_onchain}  macro: {loaded.has_macro}\n", file=sys.stderr)

    frame = features.build(loaded.frame)
    verdict = ensemble.evaluate(
        frame, fast=args.fast, run_supervised=not args.no_supervised
    )
    _print_verdict(verdict, frame)

    if args.backtest:
        print("\nFORWARD RETURNS BY OUT-OF-SAMPLE PHASE CALL (180d horizon)")
        try:
            table = ensemble.backtest_phase_calls(frame)
            print(table.round(3).to_string())
            print(
                "\n'effective_n' is overlapping windows divided by the horizon - "
                "the number of genuinely independent observations behind each row."
            )
        except Exception as exc:
            print(f"  backtest unavailable: {exc}")

    if args.pattern_audit:
        _print_pattern_audit(frame['price'].to_numpy())

    if args.position:
        _print_position(frame, verdict)

    if args.strategy:
        print("\nWALK-FORWARD STRATEGY vs BUY AND HOLD (after costs and funding)")
        try:
            result = position.backtest(frame, allow_short=args.allow_short)
            for block in (result["strategy"], result["buy_and_hold"]):
                print(
                    f"  {block['name']:<16} {block['years']:>5.1f}y  "
                    f"total {block['total_multiple']:>9.2f}x  "
                    f"ann {block['annual_log_return']:>+7.1%}  "
                    f"sharpe {block['sharpe']:>5.2f}  maxDD {block['max_drawdown']:>+7.1%}"
                )
            print(f"  average exposure {result['average_exposure']:.2f}")
            if not result["beats_hold_on_total"]:
                print(
                    "  Loses to buy-and-hold on total return. Time out of a "
                    "compounding asset is expensive; the strategy buys a "
                    "smaller drawdown with real money."
                )
        except Exception as exc:
            print(f"  strategy backtest unavailable: {exc}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(verdict.to_dict(), handle, indent=2, default=str)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
