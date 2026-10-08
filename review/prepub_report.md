# Pre-publication review — `prepub/review`

Branch `prepub/review` (from `main` @ 159f4cb). Nothing committed, pushed or called a paid API. Edited: markdown cells of nb01–nb05 and `README.md`; new file `review/prepub_report.md`. No code cell, `src/`, `scripts/`, `configs/`, `tests/` or `results/` file was touched (checked: code cells byte-identical before/after).

## A. Fresh-clone results

Clone of `main` to `/tmp`, Python 3.12.13 venv, `pip install -e ".[dev,research,llm,data]"` (18 s), no API keys in the environment.

| Step | Result |
|---|---|
| `pytest -q` | **95 passed**, 22 s, no warnings |
| nb01–nb05 `nbconvert --execute` | **all five run** once the kernel is overridden; whole cell time ≤ 1.6 s each, nothing near 60 s. Outputs match the committed outputs except one stderr line (below) |
| `run_llm_regimes.py --variant blinded --replay-only` for sonnet, gpt, gemini, glm | all four print **IDENTICAL**, 0 provider calls |
| `run_regime_bl.py`, `score_phase8.py`, `score_phase9.py` | run (18 s / 2 s / 14 s); `git status` stays clean, so every tracked result reproduces exactly |
| `score_regimes.py` (README "Reproduce") | **FAILS**: `KeyError … not in index` (`scripts/score_regimes.py:60`). It globs every `*/*_run0.csv`, which now includes the 30-month Phase 9 files `blinded_cf_{infl,growth}_run0.csv`. Nothing is written. **Blocker for the README reproduce line** |

Failures and warnings:
1. **Kernel:** every notebook has `kernelspec.name = "lra"` → `NoSuchKernel: lra` on any machine that has not registered that kernel (I used `--ExecutePreprocessor.kernel_name=python3`). Metadata also says Python 3.13.16 while `pyproject` requires `>=3.12,<3.13`.
2. nb02 cell 16: `ResourceWarning: unclosed file` (cache scan opens each file twice, never closes).
3. Jupyter prints a "kernel running over TCP without encryption" notice (harmless, environment).

Other checks: the README reproduce lines and all repo-relative links resolve. `https://github.com/FranQuant/llm-regime-allocator` returns **404 to an unauthenticated request** (private), so the five Colab badges and the clone in each Colab cell only work after the repo is public with these filenames on `main`.

## B. Findings

Status: **fixed** = fixed on this branch (markdown/README only); **needs approval** = outside the editable scope (code, metadata, configs) or needs your decision.

| File | Cell / line | Issue | Fix | Status |
|---|---|---|---|---|
| scripts/score_regimes.py | 60 | Crashes on a fresh clone (Phase 9 partial files, see A). README line removed from the reproduce chain so it is not advertised broken | skip `*_cf_*` / reindex on `y.index`; then re-add to README | **fixed** (approved; README line restored) |
| all notebooks | metadata | kernelspec `lra`, Python 3.13.16 | generic `python3` kernelspec, Python 3.12 | **fixed** (approved) |
| nb01 vs nb02 | nb01 c8 / nb02 c3 | Same forecaster, different numbers: rule Brier 1.332 vs 1.099 (0.997 vs 0.85 on the call); climatology 0.792 vs 0.770 (shift(4) approximation vs point-in-time); ML logit log-loss 1.837 vs 1.761 (clip 1e-6 vs 1e-3) | markdown now documents the difference (nb01 Setup). Code fix: nb01 calls `lra.evaluation.score/hard` and reads `regimes_fair.csv` | **fixed** (approved): nb01 now scores with `lra.evaluation`; numbers in section E |
| nb01 | whole | No Setup / Reading / Claims / Next; "context pack" vs "snapshot" | restructured to the template; added scores, playbook formula, definitions | fixed |
| nb02 | c12 | "the signature of remembered outcomes, not better reasoning" is stronger than a rolling-mean plot supports | "consistent with remembered outcomes rather than better reasoning" | fixed |
| nb02 | c17 | "the model always knew where it was": exact month 84%, not 100% | "within six months in all of them" (table: within 6m = 1.000) | fixed |
| nb02 | c21 | "2007–2012 is at random-guess accuracy" is not supported (mean error 74 months, 21% within 6 months) | removed the claim; kept "mostly lost" | fixed |
| nb02 | c34 | "no repeat-run variance yet; one model family" stale after Phase 8 | points to nb04 | fixed |
| nb02 | c34 | "Raw-pack backtests of LLM forecasters are not out-of-sample" generalised from one model | bounded to Sonnet, plus GPT via nb04 | fixed |
| nb02 | c34 | "Sonnet better on blinded than raw (paired CI excludes 0)" is not shown by any cell. I recomputed it with the notebook's helpers: −0.034, CI [−0.071, −0.004] (not written into the notebook) | add a cell or `results/` row showing it | needs approval; text marked "not displayed here" |
| nb02 | c20 output | era table row labelled `2007-12` means 2007–2012 | relabel in code | still open |
| nb03 | c7 | "about halfway between control and oracle": (0.756−0.610)/(0.961−0.610) = 42% | "about 40% of the way" | fixed |
| nb03 | c7 | κ sentence sits under the playbook table | moved to §4 | fixed |
| nb03 | c20 | "monthly turnover ~0.4–1.5× a year" garbled; table shows 0.1–1.8 one-way turnover a year | rewritten | fixed |
| nb03 | c2 | "rebalancing 2007-12 → 2026-05": last decision is 2026-04-30, held to 2026-05-29 | corrected | fixed |
| nb03 | c21 | "Next (Phase 7)" stale; "one LLM run per variant" without pointer | Next → nb04; variance pointer | fixed |
| nb03 | fig c9 | `PB`, `BL-5a/5b` abbreviations undefined in the figure; green/blue encodes significance but the same blue/green mean other things elsewhere | defined in Setup; significance now black vs grey | **fixed** |
| nb04 | c12 | "tell which month in 216 of 217 months": that is *within 12 months*, exact is 94% | reworded | fixed |
| nb04 | c18 | "so the errors are largely shared" inferred from one number | "suggests the errors overlap" | fixed |
| nb04 | c22 | "220 truncated answers": `results/llm_cache_truncated/` holds **215** files (+ `manifest.csv`, 215 rows) | replaced by your wording (220 truncated: 215 at 8k, 5 at 32k; 5 overwritten; 215 kept) | **fixed** |
| nb04 | c22 | "Every model's cutoff is after 2026-03" includes GLM, whose cutoff is unpublished | "every documented cutoff" | fixed |
| nb04 | c23 | "removing levels does not hide history from a strong reasoner" rests on one model (GPT) | bounded to GPT | fixed |
| nb04 | c20 output | portfolio table has two rows labelled `playbook__ml_gbt` (blinded 0.59 and raw 0.75) | labels now `ml_gbt (blinded inputs)` / `(raw inputs)` | **fixed** |
| nb04 | fig c6 | black marker hides the model colour | draw coloured marker on top | still open (cosmetic) |
| nb05 | c14, c18 | "they learned that strong growth precedes slowing growth" and "LLMs apply a momentum reading … one reason their edge is modest" are unsupported | ML explanation attributed to the pre-registration; LLM claim reduced to "fits the textbook reading"; modest-edge link dropped | fixed |
| nb05 | c18 | Phase 9 limit wording | states: sensitivity, not skill; edited snapshots may be read as another episode (Codex finding 5) | fixed |
| nb05 | end | no handover to live log | Next section added: live log started, **no prospective month scored**, first checkpoint after 12 scored months | fixed |
| all | figures | One 4-colour palette means regimes (nb01–02), strategies (nb01, nb03) and models (nb04–05); Sonnet is blue in nb02/04/05 but green in nb03; legends use `Risk_Off` | one model palette (Sonnet blue, GPT orange, Gemini green, GLM yellow, ML/baselines greys); regimes purple/pink/brown/navy | **fixed** (inline per notebook; shared module deferred) |
| all | code outputs | Sharpe/Brier printed with 3 decimals; style rule asks Sharpe 2, Brier 3 | tables now display Sharpe 2, Brier 3, percentages 0 (display only) | **fixed** |
| README | Finding 6 | "June–September 2026 are post-cutoff": false for Sonnet (cutoff Jun 2026); "The confirmatory test is live" implies a result | reworded; states no prospective month is scored yet, first checkpoint at 12 months, decisive result needs ~4–5 years | fixed |
| README | Finding 3 | table compares Sonnet-blinded with ML on *raw* inputs while finding 2/4 compare on *blinded* inputs | added ML-blinded row (0.871 / 0.59 / −27%, from nb02/nb03 outputs) — please vet | fixed |
| README | Finding 1 | "the signature of memory" | "consistent with memory" | fixed |
| README | Finding 4 | 2-decimal Brier ranges | 0.668–0.724 vs 0.871 | fixed |
| README | Limitations | run-on paragraph; omitted Phases 9–10 pre-registration | own section, bullets, Phase 9 limit | fixed |
| README | Reproduce | `score_regimes.py` fails (above); Python version unstated; cached responses not mentioned under licence | edited | fixed (see first row) |
| README | Models | "API spend ≈ $12 / ≈ $8 plus Ollama credits" — I cannot verify; `phase9.toml` estimates Phase 9 alone at ≈ $2.5 + $1.5 | your wording applied | **fixed** (approved) |
| configs/llm.toml | `[models.gpt]` | unrun model ID `gpt-5.5-2026-04-23` in a repo whose rule is "IDs from provider docs" | delete with `cutoffs.gpt` | needs approval |
| configs/strategy.toml | playbook | playbook/reference weights cannot be checked against the J.P. Morgan report by me | your attestation they are independent | needs approval |
| review/ | codex_*.md | internal review logs are tracked | decide whether to publish | needs approval |

## C. Code review, ranked by value (details: all items read-only, none applied)

Method notes: agent-assisted read of all `src/`, `scripts/` and notebook code cells; numbers recomputed in memory. Full text: `$CLAUDE_JOB_DIR/tmp/codereview.md` (copied below in summary).

| # | Item (location) | Proposed change | Changes numbers? | Gain |
|---|---|---|---|---|
| 1 | nb01 c8 re-implements scoring (rule 0.997, shift(4) climatology, clip 1e-6) | use `lra.evaluation.score/hard/uniform` and `regimes_fair.csv` | **Yes, nb01 table only** (rule 1.332→1.099, climatology 0.792→0.770, logit log-loss 1.837→1.761); no CSV/README/script moves | −19 lines; removes nb01/nb02 contradictions |
| 2 | `score_regimes.py:60` crash | restrict glob/reindex | No | README reproduce works |
| 3 | notebook kernelspec / 3.13 metadata | generic `python3` | No | fresh-clone execution |
| 4 | Colab cell ×5; `sys.path.insert(0,'../src')` in nb02–05 (needs cwd = `notebooks/`) | `pip install -q -e . --no-deps` on Colab; cwd-independent root | No | −25…−50 lines |
| 5 | dead code: `rules.regime_one_hot`, `[models.gpt]`/`cutoffs.gpt` + docstring in `run_llm_regimes.py`, unused `Path` in `tests/test_data_integrity.py` | delete | No | −15 lines, removes unrun model ID |
| 6 | plotting boilerplate and palettes repeated in 5 notebooks (nb05 lacks `lines.linewidth`; CI forest plot drawn 3×) | `lra/viz.py`: `apply_style`, `REGIME_COLORS` ≠ `MODEL_COLORS`, `MODEL_NAMES`, `ci_forest` | No (figures only) | −60 lines |
| 7 | loaders duplicated: `rd` in 5 scripts/notebooks (nb04's differs), macro store in 7 scripts, prices in 5, `cols()` | `lra/io.py` | No | −33 lines |
| 8 | notebooks hard-code cost 2.5 bps and `'BIL'`; `PROSPECTIVE_FROM` hard-coded in `score_live.py`/`run_live.py` although `phase10.toml` has it | read from config | No | removes silent-drift risk to the "net of costs" headline |
| 9 | `tv`, `boot_ci`, `climatology`, Phase 9 `ml_answers` live in scripts; nb01 duplicates climatology | move to `src` verbatim, add tests | No | −25 lines, closes test gap |
| 10 | nb02 c16 double-opens each cache file (ResourceWarning); nb02 c3 is a 32-line load cell | `ResponseCache.find`, `load_forecasters` | No | −22 lines, warning gone |
| 11 | `block_bootstrap_ci` 15→1.8 ms, `sharpe_diff_ci` 107→6.5 ms vectorised (bit-identical, checked) | vectorise after adding a seed-regression test | No | ~1.5 s per full run (tidiness, not speed) |
| 12 | magic numbers: `hard` 0.85, log-loss floor 1e-3, bootstrap block/n/seed, 0.05 follow threshold, 12-month "datable", blind 36/12/0.5/3.0 (and a second `36` in `context._summary`) | name them / read from registered configs; **do not change blinding constants** (cache keys) | No | clarity |
| 13 | `select_months`/`signal` row-wise apply (65 ms); `run_backtest` recomputes the cumprod per cost level (~4× possible) | vectorise / cache | No | ≤ 10 s in `run_regime_bl` (optional) |
| 14 | public API lacks docstrings; `R` means Path in nb01 and regimes elsewhere | docstrings, renames | No | readability |

Rule check against `CLAUDE.md`: no violation found (point-in-time, no date/tickers in default prompts, LLM outputs only probabilities, Σ/Ω annual with a test). One tension: `params = {}` for the LLMs is provider-default sampling, not "pinned params"; `llm.toml` says so and the cache pins outputs — suggest a one-line README note. Test gaps: TV distance, 0.05 follow/ignore thresholds, `trailing_z` values, `brier_per_obs` hand case, log-loss floor, and no test executes the notebooks.

## Licensing hygiene
- No occurrence of "Morgan"/"JPM"/"J.P." in any tracked file or in git history (`git log --all -S`); no PDF/Office files tracked. I cannot see the report, so I cannot rule out *numbers or weights* derived from it: the playbook weights in `configs/strategy.toml` are labelled "ex ante on economic grounds"; the comment "investment clock style" is a generic term. Needs your attestation.
- `data/raw/` and `.env` are git-ignored and not tracked (`git ls-files | grep -E 'data/raw|\.env$'` is empty); `.env.example` holds empty keys only; no key patterns, no `/Users/…` paths or emails in tracked files outside LICENSE.
- Consider: committed LLM responses (`results/llm_cache/`) are provider outputs; check each provider's terms; FRED third-party series (VIXCLS, BAA10Y) note is already in `data/README.md`.

## E. Changes after approval (same branch, uncommitted)

1. `scripts/score_regimes.py`: full-sample files only (skips `date_probe*`, `blinded_cf_*`, `*_p9*`, `*_every*`; date-range files never matched the glob). README reproduce line restored. Runs (rc 0); **`results/h1/scores.csv` and `calls.csv` unchanged, zero diff** (the Phase 8 rows were already in the file).
2. Metadata: `kernelspec = {name: python3, display_name: Python 3}`, `language_info = {name: python, version: 3.12}` in nb01–nb05.
3. nb01 scores with `lra.evaluation` (`uniform`, `hard`, `score`) and the point-in-time climatology from `regimes_fair.csv`; it now matches `results/h1/scores.csv`. Numbers that changed in nb01's table (nothing else in nb01 moved):

| forecaster | hit rate | Brier | log-loss |
|---|---|---|---|
| climatology | 19% → 16% | 0.792 → 0.770 | 1.764 → 1.427 |
| rule | 33% (same) | 1.332 → 1.099 | 4.809 → 2.056 |
| ML logit | 37% (same) | 0.861 (same) | 1.837 → 1.761 |
| ML GBT, uniform | unchanged | unchanged | unchanged |

   nb01 prose updated (Setup "scoring code" note, Reading: hit rates 16–37%, climatology 0.770, rule 1.099).
4. Colours: models Sonnet #2a78d6, GPT #eb6834, Gemini #1baf7a, GLM #eda100 (nb04/05 already; nb02 Sonnet lines and nb03 Sonnet now blue); ML and baselines greys; regimes (nb01, nb02) purple #7e57c2 / pink #e0559c / brown #8d6e3f / navy #1f3a5f. nb03 significance plot uses black (CI excludes 0) vs grey. Defined inline in each notebook's setup cell.
5. Display rounding via a local `fmt()` (pandas Styler) in each notebook: Sharpe 2, Brier 3, percentages 0 (hit rates, returns, volatility, drawdowns, probe shares); stored CSVs untouched. Side effect: nb04's portfolio table no longer has two identically labelled `ml_gbt` rows.
6. `scripts/archive_truncated.py`: never overwrites; a colliding record is saved as `<key>.1.json`, `.2.json`, … (tested on a temp dir; not run on the real cache). nb04 §6 now states your wording on the 220 / 215 / 5 truncated records.
7. README spend line replaced with your wording.

Checks run after the edits: `pytest` **95 passed**; replay-only for sonnet, gpt, gemini, glm all **IDENTICAL**; `score_regimes`, `run_regime_bl`, `score_phase8`, `score_phase9` all rc 0 with **no diff in `results/`**; nb01–nb05 execute end to end with the default (now `python3`) kernel, no errors; the only stderr is the known nb02 `ResourceWarning` (deferred below). The notebooks in the branch carry the freshly executed outputs, which is why the diff stat is large (embedded figures).

Still open from section B: era-table label `2007-12` in nb02 (means 2007–2012), the nb02 blinded-vs-raw CI that no cell displays (−0.034, CI [−0.071, −0.004]), nb04 marker hiding the model colour, `[models.gpt]` in `llm.toml`, playbook-weight attestation, `review/` files, provider terms for cached responses.

## After publication (deferred refactors, not done)

From section C, ranked: shared notebook helper modules (`lra/viz.py` with style, palettes, `MODEL_NAMES`, `ci_forest`; `lra/io.py` loaders; `fmt`); move `tv`, `boot_ci`, `climatology` and Phase 9 `ml_answers` into `src` with tests; cwd-independent notebooks and `pip install -e . --no-deps` Colab cell (drop `sys.path` hacks); delete dead code (`regime_one_hot`, `[models.gpt]`, unused import); read cost, cash ticker and `prospective_from` from config; fix the nb02 cache-scan `ResourceWarning` (`ResponseCache.find`); `load_forecasters`; vectorise the bootstraps and `signal`; name magic numbers (do not touch blinding constants); docstrings and renames; add tests (TV, 0.05 thresholds, `trailing_z` values, bootstrap seed regression, log-loss floor, a notebook-execution smoke test).

## F. Final scope (latest instruction: no further notebook edits)

Done in this last round, code and docs only:
1. `scripts/score_regimes.py` scores only full-sample `<variant>_run0.csv` files (skips `date_probe*`, `blinded_cf_*`, `*_p9*`, `*_every*`; date-range files never matched the glob). It runs (rc 0). **`results/h1/scores.csv` and `calls.csv`: zero diff** (no changed values and no added rows: the Phase 8 model rows were already in the file). README reproduce line includes `score_regimes.py` again.
2. `scripts/archive_truncated.py` never overwrites an archived file (`<key>.1.json`, `.2.json`, …). New `tests/test_archive_truncated.py` (2 tests: suffix logic; a second truncated record on the same cache key keeps both).
3. README spend line set to the wording given.
4. `pytest`: **97 passed** (95 + 2 new).

**Notebooks:** no notebook was touched in this round. They still carry the edits from the two earlier rounds (markdown restructure and corrections, kernelspec/Python 3.12 metadata, nb01 scoring through `lra.evaluation`, colour map, display rounding, re-executed outputs). If the rebuild should start from `main`'s notebooks instead, run `git checkout main -- notebooks/` (the README and script changes are independent of them).

## D. `git diff --stat` (current; untracked files `review/prepub_report.md` and `tests/test_archive_truncated.py` are not in the stat)
 README.md                                   |   38 +-
 notebooks/nb01_data_and_baselines.ipynb     |  512 ++++++-----
 notebooks/nb02_llm_regimes_and_memory.ipynb | 1273 ++++++++++++---------------
 notebooks/nb03_regimes_to_portfolios.ipynb  | 1049 ++++++++++------------
 notebooks/nb04_cross_family.ipynb           |  877 ++++++++----------
 notebooks/nb05_counterfactual.ipynb         |  831 ++++++++---------
 scripts/archive_truncated.py                |   15 +-
 scripts/score_regimes.py                    |    3 +-
 8 files changed, 2028 insertions(+), 2570 deletions(-)
