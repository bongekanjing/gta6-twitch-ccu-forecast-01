# NOTES — gta6-twitch-ccu-forecast-01

Entry: `main.py` at repo root. Stdlib only, no network, deterministic,
~0.2 s per case (limit 8 s). `main.ts` (starter) left untouched; the
entrypoint is `main.py`. LICENSE is the starter MIT LICENSE, untouched.

## Approach

Per-hour OLS regression of launch-day log-ratios on log hour-1 size, fitted
on the case's own rolling-midnight comparables (see `main.py` docstring for
the full write-up):

- Keep comparables whose unlock note says "rolling" / "midnight local",
  with a complete positive 24-hour launch day. In the practice cases that
  is 12 launches (ELDEN RING, Resident Evil Requiem, Hogwarts Legacy,
  Cyberpunk 2077, Ghost of Yōtei, Monster Hunter Wilds, Black Ops 6,
  MW III, MW4, Zelda: Tears of the Kingdom, FFVII Rebirth, Last of Us
  Part II). Excluded on purpose: 4 simultaneous-global launches (their
  hour 1 is pre-unlock; ratio curves explode 100–2000x), early-access
  contaminated and small/slow-burn launches, and the GTA VI category's own
  pre-release history (not a launch; its unlock note contains "rolling
  midnight local", so it is excluded by an explicit name / "not a launch"
  guard — important in the final case, where the comparables grow to 42
  and the 4 practice launches join the fit, n=16).
- `lr_h = ln(viewers_h / viewers_1)`; fit `lr_h ~ a_h + b_h * ln(p1/100k)`
  per hour. Slopes ≈ −0.4…−0.5 from hour 6 (R² 0.5–0.68 late in the day).
- `q_tau(h) = p1 * exp(mu_h + z_tau * s_h)`, `s_h = max(0.30, 0.9 * resid_sd_h)`
  plus up to +0.30 extra spread when p1 extrapolates beyond the fitted
  hour-1 range; `x = ln(p1/100k)` clamped to ln(22k/100k)…ln(1.22M/100k)
  so a record-breaking hour 1 cannot drive ratios absurd.
- Fallbacks guarantee a valid output: <3 usable comparables → per-hour
  median ratio curve; none → starter-style flat persistence. Hour-1 print
  falls back to the last history_hourly point, then 1.

## Validation (local — there is no local grade; the server grades)

Practice answers = SullyGnome hourly "Average viewers" for each launch day,
fetched from SullyGnome's chart API
(`/api/charts/linecharts/getconfig/GameViewers/{days}/{offset}/{gameid}/{name}/…`,
bare `Mozilla/5.0` UA — a full Chrome UA trips Cloudflare — plus the
per-page `Timecode` header). Verification: each answer's hour-1 value equals
the case's known print exactly (Wolverine 23,357; RDR2 116,550; GOW
Ragnarök 269,517). Game IDs: Wolverine 117671, RDR2 ("Red Dead Redemption
II") 28264, GOW Ragnarök 65645, Spider-Man 2 65106.

Pinball loss on log scale, mean over hours 2–24 × 5 quantiles
(loss = 2*max(tau*d, (tau-1)*d), d = ln(y) − ln(q)):

| case | my entry | starter | med |d| (mine) | 90% cov (mine) |
|------|----------|---------|------|------|
| p00 Wolverine (h1 23,357) | 0.186 | 0.744 | 0.336 | 1.00 |
| p01 RDR2 (h1 116,550) | 0.118 | 0.287 | 0.117 | 1.00 |
| p02 GOW Ragnarök (h1 269,517) | 0.121 | 0.215 | 0.115 | 1.00 |
| **mean** | **0.142** | **0.415** | | |
| LOO over 12 ROLL comparables | 0.246 | 0.357 | | |

(RULES.txt quotes the starter at ~0.387 on the practice; my recomputation
over p00–p02 is 0.415 — the difference is likely p03, which I cannot score.
Either way the margin is large.)

p03 (Marvel's Spider-Man 2, h1 119,087) is UNSCORED: SullyGnome's live
chart API returns all-null hourly values for that game in Oct 2023 (and
even for recent windows the game shows 0 tracked viewers); the Wayback
Machine has no archived copy of the chart-API response, and TwitchTracker
has only daily granularity for 2023. The entry runs fine on p03 (valid
output, median peak ratio ≈ 2.3× hour 1 at h21) — it is just not scored.
Do not present a p03 loss number; there is none.

Variants tested and rejected (all on practice + LOO): median-ratio curve
ignoring hour 1 (practice 0.281), ROLL + 4 extra "clean" launches (0.162),
quadratic-in-x fit (LOO 0.269), recency weighting (LOO 0.250–0.259),
Theil–Sen fit (practice 0.172), t-dist tails, sd scale 0.8–1.25 (LOO
0.243–0.252, practice 0.137–0.162 — 0.9 chosen as the middle). Channels
(viewers/channel at hour 1) showed no signal in residuals. The chosen
model is the plain per-hour OLS with sd scale 0.9.

## Risks / things to know for submit

- The final case's hour 1 could be far above the fitted range (largest
  training hour 1 is 875k; GTA VI could print 1–3M). The model clamps the
  regression input and widens the spread for extrapolation, but the
  evening peak ratio for a mega-launch is an extrapolation — the two
  biggest training launches peak at ~1× hour 1, and the model follows
  that. If GTA VI prints, say, 2M in hour 1, the forecast peak is ~0.75×
  hour 1; if the real peak is 1.5×, the loss will be dominated by that
  miss. Nothing in the data can resolve this; the clamp is the prudent
  choice.
- Worst LOO case is Resident Evil Requiem (loss 0.885): a small launch
  whose evening ramp hit 10× hour 1, far above the regression. Such
  blow-up launches exist; the spread (sd ≈ 0.45–0.5 late in the day)
  covers part of that tail but not all of it.
- The house model forecast (superbot) is the other entry to beat; its
  practice loss is unknown. My practice margin over the starter is ~66%.
- GitHub About (website https://make.fun/c/gta6-twitch-ccu-forecast-01,
  topic `makefun`) must be set at submit time — not something this repo
  can carry. README badges use owner/repo
  `bongekanjing/gta6-twitch-ccu-forecast-01`; fix if the repo name differs.
- Repo has no remote configured and was never pushed; commit on the VM
  only. The parent agent handles GitHub + submit.
