# Response to Codex adversarial review 2 (2026-10-08)

| # | Finding | Verdict | Change |
|---|---|---|---|
| 1 | `score_live.py` trusts the `prospective` flag; never checks `logged_utc` | Agree (high) | Prospective is recomputed from `logged_utc` (≤ 15 days after the decision date) and decision date ≥ 2026-10-01; disagreements with the flag are printed. |
| 2 | A changed prompt for a logged month aborts the run and blocks later months/models | Agree (high) | Logged months are no longer re-asked: their prompt hash is compared with the logged one; a mismatch is appended to `results/live/discrepancies.csv`, all other months and models still run, and the script exits non-zero at the end. Rows are never rewritten. |
| 3 | A month with exhausted parse retries stays failed forever (failed answers cached) | Agree (medium) | Invalid months move on to run ids 1, 2, … (own cache keys): cached failures are skipped for free and one fresh attempt is made per invocation, up to 5 run ids. The run id is logged. |
| 4 | `model_returned` changes are pooled, contrary to phase10.toml | Agree (medium) | `score_live.py` prints a BREAK line when `model_returned` changes, adds `model_versions` / `versions` columns, and per-version rows for the prospective sample. |
| 5 | README: "rules out pure recall" overstates Phase 9 | Agree (medium) | README finding 5 and nb05 reworded: the calls respond to the data, so they are not a pure lookup of the month; the test does not show the models stop recognising or using the month. |

Not covered by the review output (blinded-prompt date leak, portfolio lag/costs, limitations sweep): reviewed in round 1 (F1–F9); no new evidence. Tests: 95 pass.
