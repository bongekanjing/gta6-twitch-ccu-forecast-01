"""GTA VI launch-day Twitch viewership forecaster (gta6-twitch-ccu-forecast-01).

Idea
----
Every big launch's Twitch day has the same skeleton: hour 1 (00:00-01:00 UTC)
already holds the regions that unlocked at midnight local time before or at
00:00 UTC (New Zealand, Australia, Asia, Europe), and the rest of the day
ramps as the Americas unlock (US East 05:00 UTC, US West 08:00 UTC) into an
evening peak, then eases off.  What differs between launches is how much of
the global audience is already in hour 1: a launch whose hour 1 is small
relative to its total audience ramps much harder than one whose hour 1 is
already huge.

So, from the case's own comparables (past launches' hourly SullyGnome
curves), this program:

1. Keeps only genuine rolling-midnight launches: comparables whose unlock
   note says "rolling" / "midnight local", with a full 24-hour launch day
   and positive viewers every hour.  Simultaneous-global launches are
   excluded on purpose -- their hour 1 is pre-unlock (near zero), so their
   ratio curves explode 100-1000x and would poison the fit.  The target
   game's own pre-release category history is excluded too (it is not a
   launch day).
2. For each kept launch and each hour h, forms the log-ratio
   lr_h = ln(viewers_h / viewers_1).
3. Fits, separately for each hour, a one-predictor regression of lr_h on
   ln(hour1 / 100k) across the kept launches:
       lr_h ~= a_h + b_h * ln(p1 / 100k).
   The slope is about -0.4 to -0.5 from hour 6 on: the bigger hour 1, the
   flatter the rest of the day (the two biggest launches in the set, Elden
   Ring and Cyberpunk 2077, peak at roughly hour 1 itself).  The fit's
   residual spread per hour gives the forecast's spread.
4. Forecasts the target as
       q_tau(h) = p1 * exp(mu_h + z_tau * s_h),
   where mu_h is the regression prediction at the target's own hour-1
   print p1, s_h the per-hour residual spread (scaled by 0.9, floored at
   0.30 log units, plus a small widening when p1 extrapolates beyond the
   fitted hour-1 range), and z_tau the standard normal quantiles.

The anchor p1 is clamped to the fitted hour-1 range when extrapolating, so
a record-breaking hour 1 cannot drive the ratios to absurd values.

Validation (local, against SullyGnome hourly answers for the practice
launches -- see NOTES.md): mean pinball loss 0.137-0.146 on p00/p01/p02 vs
~0.415 for the starter persistence baseline; leave-one-out over the 12
rolling-midnight comparables 0.245 vs 0.357 for the starter.

Runs with the standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Regression is on ln(hour1 / 100k); these bound how far the target's hour 1
# may push the fitted line (fitted hour-1 range is ~27k..875k).
X_LO = math.log(22000.0 / 100000.0)
X_HI = math.log(1220000.0 / 100000.0)
X_FIT_HI = math.log(875000.0 / 100000.0)
SD_SCALE = 0.9
SD_FLOOR = 0.30
EXTRA_SD_PER_X = 0.15
EXTRA_SD_CAP = 0.30


def round2(x):
    """Round to 2 decimals, half up (matches JavaScript Math.round(x*100)/100)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def launch_day_hours(comp):
    """The comparable's 24 hourly viewer counts for its launch day, or None.

    Returns {hour: viewers} for hours 1..24 when the comparable has a
    complete, positive launch day; otherwise None.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for row in comp.get("hourly") or []:
        try:
            at = parse_ts(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if day_start <= at < day_start + timedelta(hours=24):
            hour = int((at - day_start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24 and isinstance(viewers, (int, float)):
                hours[hour] = viewers
    if len(hours) != 24:
        return None
    if hours.get(1, 0) <= 0 or any(v <= 0 for v in hours.values()):
        return None
    return hours


def is_rolling_midnight_launch(comp, target_name):
    """True for a genuine rolling-midnight launch comparable.

    Excludes the target game's own pre-release category history (not a
    launch), simultaneous-global launches, and anything flagged as not a
    launch.
    """
    name = (comp.get("name") or "").lower()
    unlock = (comp.get("unlock") or "").lower()
    if target_name and target_name.lower() in name:
        return False
    if "not a launch" in unlock or "target-history" in unlock:
        return False
    return "rolling" in unlock or "midnight local" in unlock


def ols(xs, ys):
    """Ordinary least squares: returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = 0.0
    sxx = 0.0
    for x, y in zip(xs, ys):
        sxy += (x - mx) * (y - my)
        sxx += (x - mx) * (x - mx)
    slope = sxy / sxx if sxx else 0.0
    return my - slope * mx, slope


def fit_ratio_model(curves):
    """Per-hour regression of log-ratio on log(hour1/100k).

    curves: list of {"h1": float, "lr": [24 log-ratios]}.
    Returns (a, b, sd): per-hour intercepts, slopes, residual sds.
    """
    a = [0.0] * 24
    b = [0.0] * 24
    sd = [0.0] * 24
    for h in range(24):
        xs = [math.log(c["h1"] / 100000.0) for c in curves]
        ys = [c["lr"][h] for c in curves]
        aa, bb = ols(xs, ys)
        dof = max(1, len(curves) - 2)
        var = sum((y - (aa + bb * x)) ** 2 for x, y in zip(xs, ys)) / dof
        a[h], b[h], sd[h] = aa, bb, math.sqrt(var)
    return a, b, sd


def median_log_ratios(curves):
    """Per-hour median log-ratio across curves (fallback when fitting fails)."""
    out = []
    for h in range(24):
        vals = sorted(c["lr"][h] for c in curves)
        n = len(vals)
        out.append(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))
    return out


def hour1_print(case):
    """Hour 1's viewer print, with fallbacks if `known` is incomplete."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and isinstance(row.get("viewers"), (int, float)):
            return max(1.0, float(row["viewers"]))
    hist = case.get("history_hourly") or []
    if hist:
        last = hist[-1]
        if isinstance(last.get("viewers"), (int, float)):
            return max(1.0, float(last["viewers"]))
    return 1.0


def forecast_case(case):
    p1 = hour1_print(case)
    target_name = ""
    game = case.get("game") or {}
    if isinstance(game, dict):
        target_name = game.get("name") or ""

    curves = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: c.get("name") or ""):
        if not is_rolling_midnight_launch(comp, target_name):
            continue
        hours = launch_day_hours(comp)
        if hours is None:
            continue
        h1 = hours[1]
        curves.append({
            "h1": float(h1),
            "lr": [math.log(hours[h] / h1) for h in range(1, 25)],
        })

    x_raw = math.log(max(1.0, p1) / 100000.0)
    x = min(max(x_raw, X_LO), X_HI)
    extra_sd = min(EXTRA_SD_CAP, EXTRA_SD_PER_X * max(0.0, abs(x_raw) - abs(X_FIT_HI)))

    rows = []
    forecast_rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    if len(curves) >= 3:
        a, b, sd = fit_ratio_model(curves)
        for row in forecast_rows:
            h = row["hour"]
            mu = a[h - 1] + b[h - 1] * x
            s = max(SD_FLOOR, SD_SCALE * sd[h - 1]) + extra_sd
            qs = [p1 * math.exp(mu + z * s) for z in Z]
            rows.append((h, qs))
    elif curves:
        med = median_log_ratios(curves)
        for row in forecast_rows:
            h = row["hour"]
            s = 0.5 + extra_sd
            qs = [p1 * math.exp(med[h - 1] + z * s) for z in Z]
            rows.append((h, qs))
    else:
        # No usable comparables at all: fall back to flat persistence.
        for row in forecast_rows:
            h = row["hour"]
            qs = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            rows.append((h, qs))

    hours = []
    for h, qs in sorted(rows, key=lambda r: r[0]):
        vals = []
        prev = 0.0
        for q in qs:
            v = round2(min(1e9, max(1.0, q)))
            if v < prev:
                v = prev
            vals.append(v)
            prev = v
        hours.append({"hour": h, "q": vals})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
