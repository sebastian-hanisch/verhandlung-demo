"""Rechnet die Messreihen der Experimente E1 bis E7 vor und schreibt data/negotiation_grid.json.

    python tools/make_grid.py            # volle Messreihen, mehrere Prozesse
    python tools/make_grid.py --quick    # kleine Probe zum Ausprobieren (data/negotiation_grid_quick.json, nicht für Tests)

Alles bis auf die Lernläufe (E7) ist deterministisch (Brüche oder Formeln); die Lernläufe benutzen SplitMix64 mit festen Zufalls-Seeds. tests/test_grid.py rechnet eine feste Probe
nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import vhd_constants as C  # noqa: E402
import vhd_evaluation as ev  # noqa: E402


def _bound_more_a():
    return ev.bound_cells(C.E2_PAIRS_MORE, (1000, 10000), (400, 401, 2000, 2001))


def _bound_more_b():
    return ev.bound_cells(C.E2_PAIRS_MORE, (10, 50, 100), (400, 401))


def _fixed_all():
    return [ev.fixed_row(d1, d2, g, rule_a) for d1, d2 in C.E2_PAIRS3 for g in (10, 20, 50, 100) for rule_a in (True, False)]


def _rule_gaps():
    return [{"d1": d1, "d2": d2, "g": g, "gap": ev.rule_gap(d1, d2, g)} for d1, d2 in ((0.9, 0.9), (0.95, 0.8), (0.5, 0.9), (0.99, 0.99)) for g in (10, 100, 1000)]


def _learn(task):
    return ev.learn_cell(*task)


def _job(name):
    return name, JOBS[name]()


JOBS = {
    "settle": lambda: [ev.settle_row(d1, d2) for d1, d2 in C.E1_PAIRS],
    "bound3": lambda: ev.bound_cells(C.E2_PAIRS3, C.E2_GS, C.E2_TS),
    "bound_more_a": _bound_more_a,
    "bound_more_b": _bound_more_b,
    "fixed": _fixed_all,
    "rule_gap": _rule_gaps,
    "rounds": lambda: {f"{r1}|{r2}": ev.short_rounds_rows(r1, r2) for r1, r2 in C.E3_RATES},
    "costs": lambda: [ev.costs_case_row(c1, c2) for c1, c2 in C.E4_CASES],
    "outside_scan": lambda: ev.outside_scan(),
    "outside_check": lambda: ev.outside_check(),
    "nash_vs_exit": lambda: ev.nash_vs_exit_rows(),
    "scale": lambda: [row for a in C.E6_AS for row in ev.scale_rows(a, 0.1)],
    "curved": lambda: [ev.curved_rubinstein_row(d1, d2, a) | {"d1": d1, "d2": d2} for a in (1.0, 0.5, 0.25) for d1, d2 in ((0.9, 0.9), (0.99, 0.99))],
    "nash_gap_short": lambda: [{"a": a, "dt": dt, "gap": ev.short_round_nash_gap(a, dt)} for a in (1.0, 0.5, 0.25) for dt in (0.1, 0.01, 0.001)],
}


def main() -> None:
    quick = "--quick" in sys.argv
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        names = list(JOBS) if not quick else ["settle", "rounds", "costs"]
        for name, value in pool.map(_job, names):
            out[name] = value
        if not quick:
            out["learn"] = list(pool.map(_learn, ev.learn_cells(), chunksize=1))
    grid = {"meta": {"learn_seeds": list(C.GRID_LEARN_SEEDS), "learn_rounds": C.GRID_LEARN_ROUNDS, "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("negotiation_grid_quick.json" if quick else "negotiation_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: {', '.join(out)}; {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
