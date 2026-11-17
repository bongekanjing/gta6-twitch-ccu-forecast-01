[![GTA VI Twitch CCU Forecast](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bongekanjing/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bongekanjing/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewership forecaster

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), hours 2–24, as five quantiles
(5%, 25%, 50%, 75%, 95%) of concurrent viewers.

## How it works

`main.py` (Python 3, standard library only, no network) reads one case file
(`.in`) and writes one forecast file (`.out`).

The forecast is anchored on hour 1's print — the count for 00:00–01:00 UTC,
which already holds every region that unlocked at midnight local time before
or at 00:00 UTC (New Zealand, Australia, Asia, Europe). The rest of the day
is predicted from the launch-day shapes of past rolling-midnight launches:

1. From the case's comparables, keep only genuine rolling-midnight launches
   (unlock note says "rolling" / "midnight local", full 24-hour launch day,
   positive viewers every hour). Simultaneous-global launches are excluded:
   their hour 1 is pre-unlock, so their ratio curves explode 100–1000x.
   The GTA VI category's own pre-release history is excluded too — it is
   not a launch day.
2. For each kept launch and each hour h, compute the log-ratio
   `ln(viewers_h / viewers_1)`.
3. Fit, per hour, a regression of the log-ratio on `ln(hour1 / 100k)`.
   The slope is about −0.4 to −0.5 from hour 6 on: the bigger hour 1, the
   flatter the rest of the day (the two biggest launches in the set, Elden
   Ring and Cyberpunk 2077, peak at roughly hour 1 itself). The fit's
   residual spread per hour sets the forecast's spread.
4. Forecast `q_tau(h) = p1 * exp(mu_h + z_tau * s_h)`, where `mu_h` is the
   regression prediction at the target's own hour-1 print, `s_h` the
   per-hour residual spread, and `z_tau` the standard-normal quantiles.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py`; the program reads every
`*.in` in the input directory and writes the matching `*.out`.

## Local validation

Scored against SullyGnome hourly answers for the practice launches
(p00 Marvel's Wolverine, p01 Red Dead Redemption 2, p02 God of War
Ragnarök — each answer's hour-1 value matches the case's known print
exactly). Mean pinball loss **0.142** vs **0.415** for the starter
persistence baseline (per case: 0.186 / 0.118 / 0.121). Leave-one-out over
the 12 rolling-midnight comparables: 0.246 vs 0.357 for the starter.

p03 (Marvel's Spider-Man 2) could not be scored: SullyGnome's live chart
API returns no hourly data for that game in October 2023, and the Wayback
Machine holds no archived copy of the API response. The entry still runs
on p03 and produces a plausible forecast (median peak ≈ 2.3× hour 1).

See [NOTES.md](NOTES.md) for the full approach and validation numbers.

## License

MIT — see [LICENSE](LICENSE).
