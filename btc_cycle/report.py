"""CLI: ``python -m btc_cycle.report``.

    python -m btc_cycle.report                    # fetch, analyse, print
    python -m btc_cycle.report --csv my_data.csv  # offline, your own export
    python -m btc_cycle.report --json out.json    # machine-readable
    python -m btc_cycle.report --backtest         # forward returns per phase
    python -m btc_cycle.report --self-test        # synthetic sanity check
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config, datasources, ensemble, features


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


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bitcoin four-year cycle phase read")
    parser.add_argument("--csv", help="offline CSV export (date,price[,mvrv,realized_price,...])")
    parser.add_argument("--json", help="write the full result to this path")
    parser.add_argument("--fast", action="store_true", help="fewer surrogates; quicker, noisier")
    parser.add_argument("--backtest", action="store_true", help="forward returns per phase call")
    parser.add_argument("--no-supervised", action="store_true")
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="synthetic sanity check, no network")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test(fast=args.fast)

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

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(verdict.to_dict(), handle, indent=2, default=str)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
