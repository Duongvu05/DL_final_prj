# MT_235 deep model vs seasonal naive and ridge. Synthesis

**Bottom line.** Both audits found no implementation bug on the data, training or evaluation path. The gap comes from the input design and the lack of a seasonal path. The wide 155-client input is a design choice, not a bug. A linear model on the same wide inputs fails the same way, with train 63.3, val 116.7 and test 100.5 kW. Run on target-only inputs, the same repo code reached test 77.8. That was one seed, hidden 64 and 12 CPU epochs. One real code defect exists in the checkpoint resume logic. It is not shown to have touched the reported numbers.

**Context the audits did not see.** The working tree already has an uncommitted redesign. It is target-only (`target_only: true`), predicts the residual over y(tau-24), uses hidden 64 and lr 5e-4, and sets Task 2 teacher forcing to a constant 1.0. The changed files are `configs/phase1.yaml`, `scripts/run_ts.py` and `src/dl_final_prj/electricity.py`. The audits judged the committed design with 159 channels. Their line numbers (run_ts.py:246, engine.py:352) do not match the current files (170 and 189 lines). I checked the resume logic again by hand. The redesign has not been trained or validated yet.

## 1. What comparable work implies

Kaggle limit. No Kaggle notebook on LD2011_2014 could be opened or found. Kaggle pages render only with JavaScript and the API needs credentials. The only Kaggle evidence is the Web Traffic 1st-place write-up and forum anecdotes. Everything else comes from GitHub repos and papers.

| Source | Setup vs ours | Key numbers | Implication |
|---|---|---|---|
| github.com/mstarckgh/ele680 | 10 clients in and out, L=168, 24 h ahead (different) | Plain GRU MAE 16.49, LSTM 17.87, TCN 16.57 vs naive_168h 14.43 and naive_24h 14.46. Residual GRU 14.49 (RMSE-noDST 27.88 vs naive_168h 28.14). Residual TCN 13.97 / 26.53 | A multi-client RNN losing to seasonal naive is reproducible. A residual target fixes it |
| github.com/FerAHMz/lab9-dl | 100 clients, global, H=24 (different) | LSTM RMSE 189.56-190.05 vs SeasonalNaive168 182.45. LSTM MASE 0.939-0.976 | Under RMSE the LSTMs lose to SN168 |
| github.com/miafuentes30/CC3092-Lab9 | 100 clients, global, H=24 | LSTM_Directa RMSE 192.93 vs SN24 228.95. DLinear 188.97, NHITS 178.77 | LSTM beats SN but loses to linear |
| github.com/timfak557/electricity-load-forecasting | MT_001, univariate, t+1 (closest in horizon) | w168 LSTM 2.1433 vs ridge 2.1481. w24 LSTM 2.1673 vs ridge 2.1992 | Even when the LSTM wins, the margin over ridge is 0.2-1.5% |
| github.com/Mike051708/BTL_HOCSAU | Same course style, univariate, t+1, no ridge | MT_162 MAE LSTM 26.632 vs SN24 96.487. MLP 20.638 beats the LSTM. Level train 293.5 / val 255.1 / test 285.0 | A univariate LSTM beats naive at t+1. A shifted val period is common |
| LSTNet, arXiv 1703.07015, Table 2 | 321 clients, pooled RSE (not comparable to per-client R2) | h=3 RNN-GRU 0.1102 vs AR 0.0995 (GRU about 11% worse). h=24 0.1295 vs 0.1054. LSTNet-Skip 0.0864 (about 13% better than AR). LRidge 0.1467 | A plain GRU loses to linear AR. Multivariate input overfits. The fix is a linear AR skip |
| PatchTST, arXiv 2211.14730, Tables 7 and 11 | ECL, T=96, z-score MSE | Channel-mixing 0.196 vs channel-independent 0.136. Instance norm 0.129 vs 0.131 | Channel mixing is the large effect. Normalization is small |
| Han et al., arXiv 2304.05206, Table 4 | ECL Transformer | CD 0.250 vs CI 0.185 | Same direction as PatchTST |
| Pelekis et al., arXiv 2302.12168 | Portuguese national aggregate, day-ahead (much smoother) | sNaive168 RMSE 402.49. Single LSTM 259.1 +- 16.4. 30-seed ensemble 213.19 | Target-only inputs plus seed ensembling. A single seed is noisy |

**Where a correct model should land on MT_235.** The current test references are SN 94.1 and ridge 79.1. Audit 1 also fit a linear AR(168) on target lags plus target-hour calendar. It scored val 84.3 and test 68.7. That probe was fit on train with alpha chosen on val, and it is not in the repo yet. The literature suggests that a correct target-only RNN with a seasonal or linear path should land between ridge and linear AR, roughly 69-79. Beating linear AR(168) by more than a few percent would be unusual, given the +0.2-1.5% and +13% margins above. A wide multi-client RNN at 103-108 is the documented failure mode, not a sign of a bug.

## 2. Confirmed implementation bugs

- **None on the data, training or eval path. Both audits agree.** Checked items were window alignment (rows o-167..o, label Y[o+h]), train-only scaling, inverse transform, BiGRU h_n indexing, eval() with no_grad, the restored best state and origin order. Audit 1 also reproduced every baseline exactly. Audit 2 sent a synthetic AR(24) series through the unchanged pipeline, and a linear model converged toward the optimum.
- **Stale checkpoint reuse (real defect, raised only by audit 2, verified by me).** `run_ts.py:71` makes `--force` skip only the metrics.json check. In `engine.py:107-118`, `fit()` loads `ckpt.pt` and returns `best_state` without training when `stopped` is set. Nothing records which npz or code version wrote the checkpoint. Before be8bb52, the calendar features were corrupted, with every stamp read as about 1970-01-15. It is unknown whether any reported 103-108 run reused an older checkpoint. The prior is low because the wide-input failure reproduces from scratch on correct data. After the redesign, an old checkpoint will raise on a shape mismatch, or silently load if the shapes happen to match. So old run directories must be deleted before reruns.
- **Already fixed earlier.** Ridge lags were 2 h off (beda1db). Timestamp storage was also wrong (be8bb52).
- **Disagreement.** Audit 1 ranks the 155-channel input as the main cause. Audit 2 ranks slow learning of the fixed lag-24 position first. Both call these modelling issues, not bugs. They do not conflict.

## 3. Ranked hypotheses for the gap

1. **Wide 155-client input causes overfitting under drift (modelling, strongest evidence).** Linear on the wide inputs gives train 63.3, val 116.7, test 100.5. That mirrors the deep runs at train about 50, val 130-146, test 103-108. Audit 1 got val z-shifts of 0.42 and test z-shifts of 0.24. *Decisive test.* Run bigru_attention with target_only=True, 3 seeds, on the pod, in fresh run directories. *If true.* Train error rises to about 72-77 kW and val drops to about 103-108. Test reaches 79 or below, near the 77.8 single-run result.
2. **No direct path to y[t+h-24] (modelling).** On synthetic AR(24), the BiGRU stalled at val 2.11-2.49 against an optimum near 1.0. Adding a residual skip gave val 1.024 at epoch 1. *Decisive test.* Run the uncommitted residual design target-only with 3 seeds. First run a zero-output sanity check. It must reproduce test SN 94.1 and val 117.8 exactly. This guards the new `inverse(pred, origins)` and `base_b24` code, which is the most likely place for a fresh bug. *If true.* Epoch-1 val is at or below 117.8, and test falls below 77.8 toward 68.7. If it stalls above 68.7, report linear AR(168) honestly as the strongest baseline.
3. **Noisy early stopping on a shifted val set, single seed (protocol).** Val is harder than test for every model. SN is 117.8 vs 94.1 and ridge is 99.7 vs 79.1. *Decisive test.* Run 5 seeds and average them, or average the last checkpoints, and compare with the best single seed. *If true.* Seed variance shrinks and test drops by a few percent. This alone cannot close a gap of more than 25%.
4. **Stale or pre-fix checkpoints in the reported runs (possible bug, low prior).** *Decisive test.* On the pod, compare each `ckpt.pt` mtime with `data/processed/electricity_hourly.npz`, and grep the logs for "resumed ... at epoch". *If true.* Some checkpoints predate the npz, and retraining changes the reported 103-108.
5. **Task 2 exposure bias in the new config (risk introduced by the redesign).** Teacher forcing is now 1.0 for all of training, while eval runs free. *Decisive test.* Compare TF 1.0 with TF 0 (free-running training) or a direct 5-output head. Score both free-running, per horizon. *If true.* RMSE for TF 1.0 grows faster with h, and the direct head matches or beats it.
6. **Distribution shift needing RevIN (low).** Evidence on Electricity shows only small gains, 0.129 vs 0.131/0.133. The residual over y(tau-24) already removes most level shift. *Decisive test.* Add RevIN to the target-only residual model. *If true.* Gain is under about 3%.
7. **DST and right-labelled calendar (negligible).** This affects all models alike. Audit 1 estimates under 1 kW.

**Report implication.** Add linear AR(168) as a baseline next to ridge, so the bar is honest at about 69. Then present target-only and residual as ablations that close the gap.

Files
- /home/vungocduong/dl_final_prj/src/dl_final_prj/engine.py (lines 107-118, resume)
- /home/vungocduong/dl_final_prj/scripts/run_ts.py (line 71, --force)
- /home/vungocduong/dl_final_prj/configs/phase1.yaml (uncommitted redesign)
- /tmp/claude-1000/-home-vungocduong-dl-final-prj/3c29501b-74ce-4dfe-b0a4-0fed46b25312/scratchpad/tscheck/real_linear.py
- /tmp/claude-1000/-home-vungocduong-dl-final-prj/3c29501b-74ce-4dfe-b0a4-0fed46b25312/scratchpad/tscheck/real_target.log