# Group 12 experiment plan

## 0. Decision summary and scope

**Task 1–2 dataset:** [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), UCI Machine Learning Repository. Use the original 370-customer, 15-minute load table, aggregate each customer to hourly mean load, and set `pred_column = MT_235`. The completed models use the target's own history plus calendar and seasonal-residual features; they do not use the 155 other clients as neural inputs.

**Task 3 dataset:** [SVHN, Format 1: Full Numbers](https://ufldl.stanford.edu/housenumbers/), not the cropped-digit format. Each original image has a `digitStruct.mat` annotation containing the ordered digit bounding boxes and labels. The target is the variable-length digit string in left-to-right order.

This pairing is deliberately distinct from the registered datasets listed in the brief: it is neither ETTh1, Hanoi/Open-Meteo weather, UCI Bike Sharing, CCPD, UFPR-AMR, nor multi-digit handwritten MNIST. The time-series task is not treated as a random IID regression problem: the central experiment is whether recurrent memory, temporal attention, selective state-space memory, and cross-customer context improve forecasts over seasonal and linear lag baselines. The image task is not treated as digit classification: it uses the full scene image and predicts the entire ordered number.

The report should make two claims only if the results support them:

1. Attention improves access to relevant daily/weekly history or spatial digit features, rather than being assumed to help.
2. The required RNN families remain competitive with strong modern comparators under a leakage-safe, repeated-seed protocol.

The main models are intentionally implementable by one student team on an RTX 3060. Tasks 1–2 are currently paused after the completed Phase 1 and Phase 1b runs; the remaining experimental effort is a focused Task 3 improvement study.

## 1. Dataset choice

### 1.1 Task 1–2: primary dataset

**Dataset:** ElectricityLoadDiagrams20112014.

**Download and provenance.** The UCI landing page was checked on 2026-11-17 and is live; its Download control provides the archive. The data are 15-minute electricity consumption measurements for 370 customers from 2011–2014. The original SVHN and UCI pages are the authoritative links to put in the registration sheet, rather than a third-party mirror. The UCI page describes the file format and variables; preserve the original archive and record its SHA-256 after download.

**Audit-informed target and preprocessing for the experiment.**

The coder's audit found that `MT_001` is exactly zero for all of 2011 and only becomes live in 2012. That would make a chronological split deceptively easy and would violate the intent of a meaningful forecasting comparison. The target is therefore permanently `MT_235`: it belongs to the full-history, low-zero group and was among the audited hardest targets for persistence (`R²=0.27` for the audited last-15%-of-live-period test). This target choice is frozen before final model training; no test metric from the final experiment may be used to change it.

- Parse the semicolon-separated, decimal-comma table and its timestamp column. Convert the 15-minute measurements to hourly means, yielding exactly 35,064 hourly timestamps from 2011-01-01 through 2014-12-31 after the audit's row convention.
- Use `MT_235` as the fixed target. The main model input is target-only: the previous 168 hourly values of `MT_235`, target-hour/calendar features, and timestamp-aligned lag features at 24, 48, and 168 hours. The train-only stable-client audit returns 155 eligible clients (the whole-series audit returns 154), but this client set is exported for audit and may be used only as a documented negative-control experiment; it is not a primary neural input.
- Add known-at-forecast-time features: `sin/cos(hour/24)`, `sin/cos(day_of_week/7)`, day-of-year sin/cos, and an explicit DST-transition flag. Do not use future observations as calendar features.
- Fit a per-target StandardScaler on the training interval only. For each target timestamp `τ=t+h`, define seasonal bases by timestamp-aligned lookup `b24(τ)=y(τ-24h)` and `b168(τ)=y(τ-168h)`, not by blindly subtracting rows from the forecast origin. Keep the UCI DST values as observed, but export DST flags and any unavailable seasonal lookup counts.
- Check timestamp regularity, duplicate timestamps, missing values, negative loads, constant columns, and zero-heavy columns before windowing. Short gaps of at most two hours may be linearly interpolated using adjacent observations; longer gaps must be removed from usable windows and counted. Never interpolate using validation or test values when constructing a training feature.
- Fit the feature standardization parameters on the training interval only. Inverse-transform predictions before all reported metrics.
- Main input window is `L = 168` hourly steps (one week). This matches common public practice while retaining the full daily/weekly context. Task 1 predicts `y[t+2]` exactly; Task 2 predicts `[y[t+1], ..., y[t+5]]`.

**Why this is a good deep-learning test.** The audited target has a large scale (`mean≈792` kW), full history, and substantially weaker persistence than the easiest customers. Public projects on this exact dataset show that daily seasonality, ridge/linear lag models, and residual learning over seasonal naive are stronger starting points than a large multi-client GRU. The Phase 1 base therefore tests the required RNN where it is methodologically plausible: target history plus calendar, with a linear/seasonal skip. The report will present these completed results honestly; no new Task 1–2 method is being added while those tasks are paused.

**Evidence-based rebasing rule.** Phase 1 is intentionally reset to the practices documented in `docs/research/electricity_research.md`: hourly aggregation, a 168-hour context, per-series train-only scaling, timestamp-aligned daily/weekly seasonal lags, and a seasonal/linear baseline before deep models. The public evidence also warns that DLinear/ridge-like models are unusually strong on this dataset, so the report must treat the full-window target-only ridge as a serious comparator and must not claim that a recurrent model is better unless its repeated-seed test result supports it. The multiclient neural input is retained only as a negative-control diagnostic because the real-data audit showed severe overfitting.

**Two alternatives.**

| Alternative | Verified link | Why it was not selected |
|---|---|---|
| PeMSD7(M) traffic speed | [STGCN dataset directory / PeMSD7_Full.zip](https://github.com/VeritasYin/STGCN_IJCAI-18/tree/master/dataset) and the direct [ZIP](https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip) | Strong spatial-temporal benchmark, but it is only about two months of weekday 5-minute data. A single-sensor RNN study would underuse the spatial structure, while a graph model would expand the project beyond the stated RNN focus. |
| NYC TLC Yellow Taxi demand | [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Freely downloadable official Parquet files and highly nonstationary demand, but it requires a substantial aggregation pipeline, very large monthly files, and careful treatment of COVID-era regime changes. It is a good fallback if UCI is unavailable, not the best few-day project choice. |

**Why the primary wins:** UCI has a stable, documented, multi-year time axis; it is small enough to preprocess once; it naturally supports both a single future point and a five-step sequence; and its multiscale load structure makes the attention hypothesis testable without requiring a graph neural network.

### 1.2 Task 3: primary dataset

**Dataset:** SVHN Format 1, Full Numbers. Use `train.tar.gz`, `extra.tar.gz`, and `test.tar.gz`; each archive contains PNG images and `digitStruct.mat`. Do not use `train_32x32.mat`, `test_32x32.mat`, or `extra_32x32.mat`, because those are cropped single-digit recognition data and do not satisfy the intended image-to-sequence task as strongly as the full-number format.

**Download and provenance.** The [official Stanford page](https://ufldl.stanford.edu/housenumbers/) was checked on 2026-11-17. It explicitly lists the full-number train, test, and extra downloads and explains that `digitStruct.mat` contains each image name and the position, size, and label of each digit bounding box. The page also states that the data are for non-commercial use; this is compatible with a university course project, but the team must not redistribute the archive or claim commercial use.

**Preprocessing for the experiment.**

- Parse each MATLAB annotation, convert MATLAB label `10` to digit `0`, sort boxes by increasing `left` coordinate, and construct the target string. Retain original image-level identity; do not generate independent digit crops as samples.
- The audit found 33,402 train images and 13,068 test images. Train lengths are 5,137/18,130/8,691/1,434/9/1 for lengths 1–6; test lengths are 2,483/8,356/2,081/146/2 for lengths 1–5. Public practice drops the single six-digit training example; do the same, set `MAX_LEN=5`, and report this exclusion explicitly. The resulting train+extra pool has 235,754 usable images before validation splitting.
- Follow the de facto Goodfellow/GitHub/Kaggle protocol for **Setting A (oracle crop)**: compute the union of all digit boxes, square it with side `max(width,height)`, expand by 30% total (`0.15*side` on each side), clip to the original image, resize to `64x64` RGB without aspect preservation, then apply a random `54x54` crop during training and a center `54x54` crop during validation/test. Normalize RGB values with mean and standard deviation `(0.5, 0.5, 0.5)`, matching the closest public attention-GRU recipe; do not use ImageNet normalization in this non-pretrained CNN base.
- **Setting B (whole image, no boxes)** is a separate trained/evaluated experiment, not a stress evaluation of Setting A. Use the complete original image without digit boxes, resize it directly to `64x64` RGB, apply the same random `54x54` training crop and center `54x54` validation/test crop, and train a separate model from scratch. Neither the crop geometry nor any input feature may use test bounding boxes. This follows the whole-image comparison style of **“Deep Learning for Scene Text Recognition” (He et al., 2019)**.
- Merge train and extra, assign a deterministic stratified 90/10 train/validation split by image hash, and keep the official 13,068-image test archive untouched. Setting A test crops are an **oracle localization protocol**: the report must say that perfect digit localization is assumed. Setting B must be reported as the no-localization result. The previously run crop-trained-checkpoint-on-whole-images score remains only a labelled cross-setting stress diagnostic and is not Setting B.
- Use full teacher forcing for the first Phase 1 base, mild `ColorJitter`, and the random crop as the main augmentation in both settings. Do not flip horizontally or use transformations that alter digit order. Preserve separate cache manifests and checkpoints for Setting A and Setting B.

**Two alternatives.**

| Alternative | Verified link | Why it was not selected |
|---|---|---|
| IIIT 5K-Word | [official IIIT/CVIT project page](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset); the page lists the 106 MB download | Real word images and transcriptions make a clean image-to-sequence task, but only 5,000 cropped word images are available and the lexicon/image protocol can make the task partly a closed-vocabulary recognition benchmark. |
| Total-Text | [official GitHub dataset page](https://github.com/cs-chan/Total-Text-Dataset) | More varied natural text and curved layouts, but it is primarily an annotated scene-text detection dataset; extracting word images and transcriptions adds substantial preprocessing and annotation-format risk for a final project. |

**Why SVHN wins:** it is a real, freely downloadable, well-documented image-to-variable-length-digit problem; the bounding boxes make the sequence order unambiguous; the train+extra pool gives enough data for a CNN+RNN model; and the cropped protocol lets the report compare honestly with the dominant Kaggle/GitHub literature while separately exposing the harder whole-image setting.

## 2. Research questions and model comparisons

### 2.1 Task 1: two-step-ahead recurrent forecast

**Question:** Does a target-history recurrent model with temporal attention improve `MT_235` prediction at `t+2` when it learns a correction to the timestamp-aligned 24-hour seasonal forecast, and does a selective state-space encoder improve that matched residual model?

**Phase 1 required model: Seasonal-residual Attentive BiGRU-2.**

- Input: 168 hourly observations of `MT_235` only, timestamp-aligned lag channels for 24/48/168 hours, and target-time calendar features. Do not pass the 155-client panel into this primary model.
- Seasonal skip: for Task 1, compute `b24(t+2)=y(t+2-24h)` by timestamp lookup. The network predicts a standardized residual `r=y(t+2)-b24(t+2)` and the final prediction is `b24+r_hat`.
- Encoder: two-layer bidirectional GRU, hidden size 64 per direction, dropout 0.20 between recurrent layers. Use the sequence of encoder outputs, not only the final state. The smaller target-only model is deliberate: public projects and the audit show that the 155-channel model overfits.
- Attention: additive/Bahdanau temporal attention with a 128-unit scoring MLP over all 168 encoder states. The context vector is concatenated with the final forward/backward state.
- Head: LayerNorm, linear 256-to-128, GELU, dropout 0.20, linear 128-to-1 residual head.
- Loss: MSE on standardized residuals; report final predictions in kW after adding back `b24` and inverse scaling. Clip no predictions.
- Optimizer: AdamW, learning rate `5e-4`, weight decay `1e-4`, gradient clipping at 1.0, batch size 256, maximum 50 epochs, early stopping patience 8 on validation RMSE, reduce learning rate by 0.5 after three stagnant epochs.
- Seeds: exactly `0, 1, 2` for every run; use the same three seeds for main models, baselines, ablations, and smoke/checkpoint tests. A smoke test may use fewer epochs, but it may not use fewer seeds and is not a final result.

The design is SOTA-informed but requirement-compliant. It uses the recurrent sequence encoder and attention pattern established for sequence-to-sequence learning, while the known calendar covariates and interpretable temporal weighting are informed by **“Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting” (Lim et al., 2021)**. Comparators include **“N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting” (Oreshkin et al., 2020)**, **“Are Transformers Effective for Time Series Forecasting?” (Zeng et al., 2023)**, and **“A Time Series is Worth 64 Words: Long-term Forecasting with Transformers” (Nie et al., 2023)**. The report will present the completed recurrent comparison with repeated seeds and formal paired tests; no additional Task 1 method is planned while the task is paused.

**Baselines and SOTA-informed comparators.** Tasks 1–2 are paused with the following completed comparison set:

1. Persistence: `y[t+2] = y[t]`.
2. Timestamp-aligned seasonal naive: `y[t+2] = y[t+2-24h]` and `y[t+2] = y[t+2-168h]`.
3. Corrected sparse ridge on lags `{1, 2, 3, 24, 48, 168}` plus calendar features.
4. Full-window target-only ridge on all 168 target lags plus target-time calendar features; this is the linear diagnostic that reached about 68.7 kW RMSE.
5. Target-only GRU-2 without attention, with the same `b24` residual skip.
6. Phase 1 required Seasonal-residual Attentive BiGRU-2.
7. The 155-client GRU is a documented negative-control experiment only, not a headline baseline. N-BEATS, DLinear, and PatchTST remain literature context and are not locked runs.

The primary course comparison is the Attentive BiGRU versus the attention-free target-only GRU and best non-neural baseline. The report must not imply that the recurrent model beats the full-window ridge when the paired test does not support that claim.

### 2.2 Task 2: five-step encoder–decoder forecast

**Question:** Does a target-only residual encoder–decoder with attention improve the five timestamp-aligned horizons over seasonal and linear baselines?

**Phase 1 required model: Seasonal-residual Bahdanau Attentive LSTM Encoder–Decoder.**

- Input: the same target-only 168-hour history, timestamp-aligned lag features, and calendar features as Task 1; output length exactly `H=5`.
- Seasonal skip: for each horizon `h`, compute `b24(t+h)=y(t+h-24h)` by timestamp lookup. The decoder predicts residuals `r_h=y(t+h)-b24(t+h)` and the final output is `b24+r_hat`.
- Encoder: two-layer LSTM, hidden size 64, dropout 0.20. Retain all encoder outputs and the final hidden/cell states.
- Decoder: one-layer LSTM, hidden size 64, initialized from the encoder final states. At each horizon, input the previous residual, known future calendar embedding, and a learned horizon embedding for `1..5`.
- Attention: additive attention over all encoder outputs at every decoder step. Report the five attention maps.
- Output: one residual scalar per decoder step. Train with teacher forcing ratio 1.0 in Phase 1 to match public sequence-to-sequence practice; at validation/test use fully autoregressive decoding. A scheduled-sampling sensitivity is deferred.
- Loss: mean MSE over the five standardized residuals, with equal horizon weights. Report all metrics after adding back the five seasonal bases and inverse transformation.
- Optimizer: AdamW, learning rate `5e-4`, weight decay `1e-4`, gradient clipping 1.0, batch size 128, maximum 60 epochs, patience 10, and exactly seeds `0, 1, 2`.

This model follows **“Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation” (Cho et al., 2014)** and **“Neural Machine Translation by Jointly Learning to Align and Translate” (Bahdanau et al., 2015)** for recurrent encoder–decoder and learned alignment, then adapts them to multi-horizon load forecasting. It is informed by **“Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting” (Lim et al., 2021)** because it conditions each future horizon on known covariates and analyzes temporal importance. The strongest modern comparators are N-BEATS (2020), DLinear (2023), and PatchTST (2023). Do not call the LSTM itself SOTA; call it an interpretable, requirement-compliant recurrent model tested against SOTA-informed baselines. Task 2 is paused after the completed Phase 1 comparison.

**Baselines and comparators.**

- Five-step persistence, timestamp-aligned daily seasonal, and weekly seasonal naive forecasts.
- Corrected sparse ridge and full-window target-only ridge direct multi-output regressors.
- LSTM residual encoder–decoder without decoder attention.
- Phase 1 required Seasonal-residual Attentive LSTM encoder–decoder.
- The old 155-client LSTM and N-BEATS/DLinear/PatchTST remain diagnostics/literature context, not headline runs.

Report both pooled metrics and metrics separately at horizons 1–5. The key failure mode to analyze is a model that has a good average but deteriorates sharply at horizon 5.

### 2.3 Task 3: CNN encoder plus recurrent attention decoder

**Question:** How much can a single CNN encoder plus attentive GRU decoder improve ordered variable-length house-number recognition in both the oracle-crop and no-localization settings?

**Phase 1 required model: Goodfellow/q19316-style SVHN CNN + attentive GRU decoder, trained separately in Settings A and B.**

- Encoder: VGG-BN-style CNN with four convolutional stages, batch normalization, ReLU, pooling, and dropout, producing approximately `7x7` spatial tokens from the `54x54` crop. This is closer to the public SVHN attention-RNN practice than a ResNet-18 whose feature map becomes too small after aggressive downsampling.
- Convert the feature map to visual tokens with learned 2-D position embeddings. Decoder: two-layer GRU, hidden size 256, additive visual attention over the tokens at every character step. Vocabulary is digits 0–9 plus `<SOS>`, `<EOS>`, and `<PAD>`.
- Add a lightweight auxiliary length head with five classes (length 1–5), cross-entropy weight 0.2. The official GRU result is beam-decoded; greedy and length-constrained greedy outputs are diagnostics only.
- Training: full teacher forcing, cross-entropy without label smoothing, AdamW at `3e-4`, weight decay `1e-4`, batch size 64, maximum 16 epochs, learning rate multiplied by 0.1 at epoch 10, gradient clipping 5.0, and early stopping on validation exact-string accuracy.
- Train separate Setting A and Setting B models on the same 90% train+extra split using random `54x54` crops and moderate `ColorJitter`; evaluate each with center `54x54` crops. For each GRU decoder and setting, select one beam width from `{3, 5, 10}` using mean validation exact-match accuracy across seeds `0,1,2`, after each checkpoint has been selected by validation greedy exact match; break validation ties in favor of the smaller beam. Freeze that width and apply it unchanged to all three seeds and test. Report beam decoding, never greedy, as the official GRU result. Setting A is labelled oracle localization, while Setting B uses no boxes at train or test time.
- For CRNN-CTC, select one CTC prefix-beam width from `{3, 5, 10}` using mean validation exact match across the three seeds and freeze it before test. For the multi-head CNN, use exact MAP decoding over length and digit-head probabilities, with no test-label length constraint. No decoding width, checkpoint, or post-processing rule may be selected from test results.

**Task 3 decoding hierarchy and table names.** The report has one primary decoding per model and three secondary GRU decodings. The primary rows are `beam_primary` for both GRU decoders, `ctc_prefix_primary` for CRNN-CTC, and `multihead_map_primary` for the multi-head CNN. `beam_primary` constrains hypotheses to digit tokens followed by EOS and at most `MAX_LEN=5` digits; its width is selected from `{3,5,10}` on validation. The checkpoint itself remains selected by validation exact match under greedy decoding, as pre-registered; the selected checkpoint is then held fixed while decoding widths and rescoring rules are selected on validation.

- `mbr_edit_secondary`: edit-distance minimum Bayes-risk decoding over the frozen primary-beam n-best list, using the normalized beam posterior and choosing the candidate with minimum expected Levenshtein distance. It has no test-tuned parameter. This follows **“Minimum Bayes-Risk Decoding for Statistical Machine Translation” (Kumar and Byrne, 2004)**.
- `length_rescore_secondary`: choose one setting/model-level `lambda_length` on validation from `{0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}` by mean validation exact match across seeds, score each beam candidate as `log P_seq + lambda_length * log P_length`, and apply the frozen lambda unchanged to all three seeds and test images.
- `dram_fb_secondary`: for the main attentive GRU only, train a same-architecture right-to-left decoder on reversed target strings and score each left-to-right beam candidate as `log P_L2R + lambda_fb * log P_R2L(reverse(candidate))`. Select one setting-level `lambda_fb` from the same validation grid by mean validation exact match across seeds, freeze it, and apply it unchanged to all seeds and test images. This is the forward-backward idea of **“Multiple Object Recognition with Visual Attention” (Ba et al., 2015)** (DRAM).

MBR and length-head rescoring are reported for both GRU models in Settings A and B. DRAM-style rescoring is reported only for the main attentive GRU in both settings, using six additional right-to-left training runs (three seeds per setting). Greedy, unconstrained length-head greedy, and any unselected beam widths remain labelled diagnostics and cannot be used as headline results. Exact constrained CTC sequence MAP is an optional decoding-only verification row, `ctc_seqmap_diagnostic`; it is not a primary result and adds no training runs.

This is SOTA-informed by **“Show, Attend and Tell: Neural Image Caption Generation with Visual Attention” (Xu et al., 2015)**, which established spatial attention for image-to-sequence generation; **“An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition” (Shi, Bai, and Yao, 2017)**, which established the CNN–recurrent sequence-recognition pattern; and **“ASTER: An Attentional Scene Text Recognizer with Flexible Rectification” (Shi et al., 2018)**, an attention-based scene-text sequence recognizer. A full ASTER or ABINet reimplementation is not required. The project contribution is a clean, auditable comparison between visual attention, CTC, and pooled recurrent decoding under the course’s four regression metrics, plus sequence accuracy/edit distance as necessary task metrics.

**Baselines and comparators.**

1. In each setting, Phase 1 CNN + attentive GRU with the auxiliary length head, reported with validation-selected beam search.
2. In each setting, the same CNN + GRU decoder without visual attention, also reported with validation-selected beam search.
3. In each setting, the public-practice multi-head CNN length/position classifier with exact MAP decoding.
4. In each setting, the CRNN-CTC diagnostic baseline with validation-selected prefix beam search; report its protocol caveat honestly.
5. The old crop-trained-checkpoint-on-whole-image evaluation remains a labelled cross-setting stress diagnostic, not Setting B. PARSeq remains literature context and is not a locked run.

Primary image metrics are computed on the decoded sequence, not on teacher-forced logits. Also report exact full-string accuracy, character accuracy, normalized Levenshtein distance, and length accuracy; these are necessary because four numeric regression metrics do not fully describe OCR quality.

### 2.4 Phase 1 results and the Phase 1b linear-highway analysis

The completed three-seed Phase 1 results are final for Tasks 1–2 for now. Task 1 test RMSE is persistence `199.5` kW, daily seasonal `94.1`, weekly seasonal `105.9`, sparse ridge `79.1`, full-window ridge `68.78`, residual BiGRU `70.02±0.35`, and attentive residual BiGRU `70.68±0.76` (`R²≈0.909`). Task 2 pooled RMSE is daily seasonal `94.1`, sparse ridge `79.1`, full-window ridge `70.76`, residual LSTM encoder–decoder `73.45±1.38`, and attentive residual encoder–decoder `73.66±0.85`. Deep models beat full-window ridge on validation but not on the held-out test regime, so the report must not claim that attention or recurrence beats ridge. The Task 1 deep-versus-daily DM result is significant (`p=0.007`), while deep-versus-full-window-ridge and attention-versus-no-attention are not significant.

The user-requested full-window linear highway is a documented **Phase 1b supplementary improvement**, not a replacement for the registered Phase 1 models and not a new claim. It is the LSTNet-style jointly trained linear skip over the standardized target window and target-hour features. The attentive highway model gives Task 1 `69.30±0.44` kW across seeds; averaging the three seed forecasts gives `68.49`, and validation-selected blending with full-window ridge (`0.9` highway ensemble + `0.1` ridge) gives `67.60`. For Task 2, the corresponding values are `72.37±1.12`, `70.60`, and `69.51` with validation-selected weights `0.8/0.2`. The Task 1 blend versus full-window ridge has bootstrap difference `-1.18` kW with 95% CI `[-6.02,+3.96]` and DM `p≈0.6`; therefore these are promising point estimates, not statistically established wins. Report them in a separate Phase 1b table and use the full-window ridge as the primary non-neural reference. Tasks 1–2 are paused after this honest comparison; do not add new methods or rerun them until explicitly reopened.

The completed Task 3 primary results are validation-selected non-greedy results: Setting A attention GRU `95.72%±0.17`, pooled GRU `95.54%±0.11`, multi-head CNN `95.16%±0.05`, and CRNN-CTC `96.00%±0.14`; Setting B attention GRU `71.76%±0.11`, pooled GRU `72.03%±0.03`, multi-head CNN `68.85%±0.23`, and CRNN-CTC `71.98%±0.11`. The main attention GRU's DRAM secondary decoding is `96.14%±0.11` in A and `73.20%±0.05` in B, with validation-selected lambdas 2.0 and 1.0. The old crop-trained-checkpoint-on-whole-image result (`19–24%`) remains only a labelled stress diagnostic, not Setting B. New work is limited to the improvement variants below.

## 3. Task 3 improvement study

Every completed ablation uses the same split, preprocessing, optimizer budget, early-stopping rule, and checkpoint policy as its parent model. Tasks 1–2 are paused: their tables and Phase 1b linear-highway analysis are final for now. New work is only the Task 3 improvement study below, with exactly seeds `0, 1, 2`.

### Completed Task 1 ablation record (no new runs)

| ID | Variant | What it tests |
|---|---|---|
| T1-P0 | Phase 1 seasonal-residual Attentive BiGRU-2, target-only input | Required course model on the public-practice base |
| T1-P1 | Same residual model without temporal attention | Required no-attention counterpart |
| T1-P2 | Same model without the `b24` residual skip | Diagnostic: value of residual learning; not a Phase 1 gate |
| T1-P3 | Same model with 155-client panel input | Negative control for the observed overfit; not a headline model |
### Completed Task 2 ablation record (no new runs)

| ID | Variant | What it tests |
|---|---|---|
| T2-P0 | Phase 1 seasonal-residual attentive LSTM encoder–decoder | Required course model on the public-practice base |
| T2-P1 | Same residual encoder–decoder without decoder attention | Required no-attention counterpart |
| T2-P2 | Same model without the `b24` residual skip | Diagnostic: value of residual learning |
| T2-P3 | Same model with 155-client panel input | Negative control for the observed overfit |
### Task 3 improvement ablations

| ID | Variant | What it tests |
|---|---|---|
| T3-P0 | Setting A or B CNN + visual-attention GRU + length head | Required course model in each input setting |
| T3-P1 | Same setting-specific model without visual attention | Required no-attention counterpart |
| T3-P2 | Setting-specific multi-head CNN length/position classifier | Common public-practice non-recurrent baseline |
| T3-P3 | Setting-specific CRNN-CTC | CTC diagnostic with prefix-beam decoding |
| T3-P4 | Crop-trained checkpoint evaluated on whole images | Cross-setting localization/scale diagnostic only |
| T3-I0 | Frozen `cnn_attention_gru` (6,202,898 parameters), validation-selected constrained beam | Reference in A and B; already run with seeds 0,1,2 |
| T3-I1 | Same model with DRAM forward–backward rescoring; RTL decoder trained on reversed strings | Primary decoder improvement; already run in A and B |
| T3-I2 | RandomResizedCrop(54, scale=(0.70,1.00)) from the 64-pixel canvas, replacing translation-only crop | Scale-robustness ablation in A and B |
| T3-I3 | Character and length-head cross-entropy label smoothing 0.05 | Secondary robustness/label-noise ablation in A and B |
| T3-I4 | Identity-initialized affine spatial transformer before VGG-BN, output 64x64, no boxes | Primary geometry/localization ablation in B only |
| T3-I5 | Whole-image resize 96x96, random 80x80 train crop, center 80x80 eval crop | Primary resolution ablation in B only |
| T3-I6 | Pre-registered B combination of I2+I3+I4+I5 | Final practical candidate; secondary combined claim |

The study is ordered: lock the largest observed decoder gain first, then test one change at a time against I0, then run the combined B recipe. It is motivated by **“Multiple Object Recognition with Visual Attention” (Ba et al., 2015)** for forward–backward rescoring, **“Spatial Transformer Networks” (Jaderberg et al., 2015)** for learned rectification, **“Deep Learning for Scene Text Recognition” (He et al., 2019)** for whole-image recognition, and the public SVHN attention-RNN practice documented in `docs/research/svhn_protocol_research.md` for the 64/54-pixel and scale-augmentation recipe. Existing pooled-GRU, multi-head, and CTC results remain Phase 1 comparators and receive no new training.

The primary improvement claims are I1 versus I0 in each setting, I2 versus I0 in each setting, I4 versus I0 in B, and I5 versus I0 in B. I3 and I6 are secondary unless their validation and paired test results support a clearly stated claim. Every single-factor run changes only the named component; keep the base optimizer, epochs, batch size, checkpoint rule, decoder, and parameter budget fixed. For I4/I5 export the token-map geometry and exact parameter count.

**Exact recipe for new runs.** Retain VGG-BN channels `(64,128,256,512)`, two-layer GRU hidden 256, Bahdanau attention over visual tokens, length loss weight 0.2, AdamW `3e-4`, weight decay `1e-4`, batch 64, 16 epochs, StepLR x0.1 after epoch 10, gradient clip 5, bf16, full teacher forcing, and ColorJitter 0.3. Use the same 212,180/23,574 train/validation images and hash split. I2 changes only the crop sampler; I3 sets label smoothing to 0.05 for character and length CE; I4 adds an identity-initialized affine STN with bilinear sampling and no box labels; I5 uses a separate 96-pixel cache and changes only the crop geometry. Checkpoint on validation greedy exact match, but never report greedy.

### 3.1 Decoding and evidence rules

`beam_primary` remains the fair primary decoder for the main GRU. `dram_fb_secondary` is the pre-registered best improved decoder and is the headline “with forward–backward rescoring” row when it wins on validation; both names remain in every table. Existing A results are 95.72%±0.17 beam and 96.14%±0.11 DRAM; B results are 71.76%±0.11 beam and 73.20%±0.05 DRAM. MBR-edit and length-head rescoring remain secondary rows; CTC prefix beam and multi-head exact MAP remain Phase 1 comparator protocols.

For every new variant, select the checkpoint by validation greedy exact match, select beam width from `{3,5,10}` on validation across all three seeds, freeze it, and apply it unchanged to all seeds and test. Export the selected width and any DRAM lambda. Use a 10,000-replicate paired image bootstrap, averaging same-seed per-image differences before resampling, for exact accuracy, character accuracy, MAE, RMSE, and normalized edit distance. Use McNemar only as a descriptive per-seed check. Apply Holm correction within each setting to the four primary improvement families; I3/I6 are secondary. With three seeds, do not use a seed-level t-test or treat seed-image pairs as independent.

## 4. Evaluation protocol and leakage controls

### 4.1 Splits

**Time series:** chronological 70% train, 15% validation, 15% test by hourly timestamp. Construct a window only when the forecast target lies wholly in its split. It is acceptable for a validation/test input window to use historical context immediately before the split boundary, because that is available at prediction time; it is not acceptable to use any future value. Fit scalers, feature selection, missing-value rules, and all hyperparameters using train/validation only. Test is run once after model selection.

**SVHN:** remove the single six-digit train example, merge the remaining official train and extra images, and split the resulting 235,754-image pool 90/10 into train/validation with a fixed hash manifest stratified by sequence length. Keep the official 13,068-image test archive untouched. Train and evaluate two separate settings: Setting A crops every image with its own GT union box expanded 1.3x and is labelled oracle localization; Setting B uses the complete image resized to `64x64` with no boxes at train or test time. The old crop-checkpoint-on-whole-image result is only a stress diagnostic. Any duplicate-image audit must happen before splitting; the completed audit found no exact SHA-1 duplicates, but found near-duplicate groups requiring manual review, including one train/test group. Resolve or explicitly report any confirmed cross-split near-duplicate before final metrics.

### 4.2 Reproducibility and preemption

- Final seeds are locked by the matrix in Section 4.4: exactly seeds `0, 1, 2` for every model and ablation. Report mean, standard deviation, median, and every seed; rank models by mean test RMSE or exact-sequence accuracy, not by the best seed.
- Set Python, NumPy, PyTorch, and data-loader seeds; use deterministic cuDNN where practical; record GPU type, software versions, commit hash, dataset checksum, and exact command line.
- Save the best validation checkpoint, optimizer state, scheduler state, scaler, RNG states, current epoch, test predictions, per-origin/per-horizon errors, attention weights, and split manifest. Never select a checkpoint using test metrics.
- Standardize on the training interval only, and compute all metric values after inverse transformation. Store both scaled loss and physical-unit metrics.

The RunAI pod is preemptible, but the full 40 GB A100 is now available for this study. Use mixed precision, checkpoint at the end of every epoch and every 1,000 optimizer steps, and resume rather than restart after eviction. Record GPU type, software versions, commit hash, dataset checksum, and exact command line.

### 4.3 Statistical tests

Tasks 1–2 retain their already-computed paired tests and are not reopened. New confirmatory tests concern Task 3 improvement variants versus the frozen `cnn_attention_gru` reference.

- Use the Diebold–Mariano test on paired per-origin losses. For Task 1 use squared error and absolute error separately; for Task 2 use the average five-horizon squared error per origin and additionally test each horizon. Because rolling origins overlap, use a Newey–West/HAC variance with truncation lag `H-1` (or a conservative 24-hour lag for hourly forecasts) and report two-sided p-values with a 5% threshold.
- Use a paired moving-block bootstrap with 10,000 replicates over chronological test origins, block length 168 hours. Report the 95% confidence interval for the difference in RMSE, MAE, and pooled NSE. Do not resample individual overlapping rows independently.
- For the frozen Task 1/2 results, retain the per-origin DM and block-bootstrap outputs already exported. For Task 3, average the same-seed per-image difference across seeds `0, 1, 2` before the paired bootstrap, report each seed's result, and apply Holm correction within each setting to the four primary improvement families (I1, I2, I4, I5). I3 and I6 are secondary.

For SVHN, use a paired bootstrap over test images with 10,000 replicates for the difference in exact-string accuracy, character accuracy, MAE, RMSE, and normalized edit distance, separately for Setting A and Setting B. For exact-string correctness, add McNemar's test between each improvement and I0 as a descriptive secondary check. Use two-sided tests and state that significance is secondary to effect size and confidence intervals.

With exactly three seeds, the primary image test is the paired bootstrap over per-image differences after averaging the three same-seed correctness/loss values. Do not pretend that the 3 x 13,068 seed-image pairs are independent. Decode-width and rescoring selection is performed only on validation: GRU and CTC prefix beams test widths `{3,5,10}`, secondary lambdas use the pre-registered grid, mean validation exact match is pooled across the three seeds, ties choose the smaller width/lambda, and the chosen setting is frozen before test. Multi-head MAP decoding has no tunable width. McNemar's test may be reported separately for each seed as a descriptive secondary check, but the claim-supporting interval is the clustered per-image bootstrap; no seed-level t-test is allowed.

Do not claim that a p-value proves one deep architecture is universally superior. The valid claim is conditional on this dataset, horizon, preprocessing, and model budget.

### 4.4 Locked final run matrix and A100 budget

This is the final matrix, not a menu. A short smoke/checkpoint run is allowed before it, but it must also use exactly seeds `0, 1, 2` and cannot replace the listed runs. Every listed model/baseline evaluation uses exactly seeds `0, 1, 2`. Tasks 1–2 are completed and paused; only the Task 3 improvement rows below are new work. The modern N-BEATS/DLinear/PatchTST papers remain literature context, not locked runs.

| Phase | Task/setting | Model family | Seeds | Runs | Estimated active A100 hours |
|---|---:|---|---:|---:|---:|
| 1 | 1 | Persistence, daily/weekly timestamp-aligned seasonal naive, sparse ridge, full-window target-only ridge | 0,1,2 | 12 model/seed evaluations | 0.0 |
| 1 | 1 | Seasonal-residual Attentive BiGRU-2 and same residual GRU without attention | 3 each | 6 | 0.8 |
| 1 | 2 | Persistence, daily/weekly timestamp-aligned seasonal naive, sparse ridge, full-window target-only ridge | 0,1,2 | 12 model/seed evaluations | 0.0 |
| 1 | 2 | Seasonal-residual attentive LSTM encoder–decoder and same residual decoder without attention | 3 each | 6 | 0.9 |
| 1 | 3A oracle crop | Original train+extra CNN-attention GRU, no-attention GRU, multi-head CNN, and CRNN-CTC training | 3 each | 12 model/seed evaluations, completed Phase 1 run | 15.6 already spent |
| 1 | 3A oracle crop | Existing cropped train+extra CNN-attention GRU, no-attention GRU, multi-head CNN, and CRNN-CTC checkpoints; re-evaluate with validation-selected beam/MAP decoding | 3 each | 12 decoding evaluations, no retraining | 0.5 |
| 1 | 3B whole image | Separately trained whole-image train+extra CNN-attention GRU, no-attention GRU, multi-head CNN, and CRNN-CTC | 3 each | 12 model/seed evaluations | 15.6 |
| 1 | 3A + 3B | Right-to-left attentive-GRU decoder for DRAM-style rescoring of the main attention GRU, one model per setting | 3 each | 6 completed decoder model/seed evaluations | 2.0 already spent |
| 1 | 3 diagnostic | Crop-trained checkpoint evaluated on whole-image inputs | 3 | 3 cross-setting evaluations | 0.3 |
| 1 | all | Three-seed smoke runs, extra-cache/checkpoint validation, and Phase 1 export verification | 0,1,2 | operational checks, not model/seed totals | 1.5 |
| 1 | 3A + 3B | I2 random-resized scale augmentation, main CNN-attention GRU | 3 each | 6 new model/seed evaluations | 2.5 |
| 1 | 3A + 3B | I3 label smoothing 0.05, main CNN-attention GRU | 3 each | 6 new model/seed evaluations | 2.5 |
| 1 | 3B whole image | I4 affine spatial transformer, main CNN-attention GRU | 3 | 3 new model/seed evaluations | 1.25 |
| 1 | 3B whole image | I5 96x96/80x80 higher-resolution input, main CNN-attention GRU | 3 | 3 new model/seed evaluations | 1.25 |
| 1 | 3B whole image | I6 combined I2+I3+I4+I5 recipe | 3 | 3 new model/seed evaluations | 1.25 |
| 1 | all | Checkpoint/resume smoke, cache validation, decoding/statistics/export verification | 0,1,2 | operational checks, not model/seed totals | 1.0 |
| **Total new** |  |  |  | **21 new trained model/seed evaluations** | **9.75 active hours** |

CPU-only Tasks 1–2 metrics, Phase 1b blending, statistical tests, and report exports do not consume meaningful A100 time. The completed Phase 1/decoder results are retained; the new Task 3 allowance is **9.75 active A100 hours** for 21 individual seed runs, or about **3.5–4.5 wall-clock hours** when the three seeds are launched in parallel, plus preemption/resume margin. Reserve **15 wall-clock hours**. Do not add alternative OCR architectures, extra hyperparameter searches, or unregistered training changes before the report is complete.

## 5. Exact metrics

Let `y` denote the original-unit target and `hat{y}` the original-unit prediction. Use the same denominator convention for R2 and NSE so the relationship between them is transparent.

### 5.1 Single-step Task 1

For test origins `i = 1..N`, with target `y_i = y[t_i+2]` and prediction `hat{y}_i`:

```text
MAE  = (1/N) * sum_i |hat{y}_i - y_i|
RMSE = sqrt((1/N) * sum_i (hat{y}_i - y_i)^2)
R2   = 1 - sum_i (hat{y}_i - y_i)^2 / sum_i (y_i - mean(y))^2
NSE  = 1 - sum_i (hat{y}_i - y_i)^2 / sum_i (y_i - mean(y))^2
```

Here `mean(y)` is the test-target mean computed over the evaluated test origins. Therefore R2 and NSE are numerically identical under this convention; report both because the assignment requires both, and explicitly note the identity. A constant-persistence or train-mean prediction may have negative R2/NSE; do not clip values.

### 5.2 Multi-horizon Task 2

Let `y_{i,h}` and `hat{y}_{i,h}` be the true and predicted values for origin `i` and horizon `h in {1,...,5}`. The **primary** result is pooled over all `N*5` forecast cells:

```text
MAE  = (1/(5N)) * sum_i sum_h |hat{y}_{i,h} - y_{i,h}|
RMSE = sqrt((1/(5N)) * sum_i sum_h (hat{y}_{i,h} - y_{i,h})^2)
R2   = 1 - sum_i sum_h (hat{y}_{i,h} - y_{i,h})^2
           / sum_i sum_h (y_{i,h} - mean_all(y))^2
NSE  = 1 - sum_i sum_h (hat{y}_{i,h} - y_{i,h})^2
           / sum_i sum_h (y_{i,h} - mean_all(y))^2
```

`mean_all(y)` is the mean of all true test forecast cells. This gives every forecast cell equal weight and avoids giving longer sequences more weight than others. Also report the same four metrics separately for each horizon and the arithmetic mean of the five horizon-specific metrics. The horizon-specific R2/NSE denominators use that horizon’s true mean; do not compare a horizon-specific R2 to the pooled R2 as if they were identical.

Two defensible secondary conventions should be mentioned in the report: (1) macro-average the metric computed separately for each origin, and (2) compute metrics per horizon and average them. The primary pooled convention is preferable for the headline result because it is stable and directly evaluates all five required outputs.

### 5.3 Image-to-sequence Task 3

For image `i`, let the ground-truth digit sequence be `y_i = (y_{i,1},...,y_{i,L_i})`, with `y_{i,j} in {0,...,9}`. Decode with the frozen validation-selected beam/MAP rule, remove `<SOS>`, stop at `<EOS>`, and let `hat{y}_{i,j}` be the decoded digit at position `j`.

To prevent extra digits from being invisible to the required numeric metrics, the **primary numeric convention is a penalty-aware variable-width representation**. Reserve numeric sentinel `10` for a missing or extra sequence position. For each image set `q_i=max(L_i,hat{L}_i)`, `u_{i,j}=y_{i,j}` when `j<=L_i` and `10` otherwise, and `v_{i,j}=hat{y}_{i,j}` when `j<=hat{L}_i` and `10` otherwise, for `j=1..q_i`. Thus an early EOS creates a prediction of 10 against a true digit, and an extra digit creates a prediction of 0–9 against a true sentinel 10. Do not include positions after `q_i` where both sides are absent.

Let `M = sum_i q_i` and let `v_{i,j}` and `u_{i,j}` be the aligned values above:

```text
MAE  = (1/M) * sum_i sum_{j=1..q_i} |v_{i,j} - u_{i,j}|
RMSE = sqrt((1/M) * sum_i sum_{j=1..q_i} (v_{i,j} - u_{i,j})^2)
R2   = 1 - sum_i sum_{j=1..q_i} (v_{i,j} - u_{i,j})^2
           / sum_i sum_{j=1..q_i} (u_{i,j} - mean(u))^2
NSE  = 1 - sum_i sum_{j=1..q_i} (v_{i,j} - u_{i,j})^2
           / sum_i sum_{j=1..q_i} (u_{i,j} - mean(u))^2
```

This is the primary convention because it penalizes both early termination and extra digits rather than silently dropping them. The sentinel is a pre-registered evaluation value, not a model input or training label. R2 and NSE are identical under the shared denominator. Also report these secondary sequence conventions:

1. **Position macro-average:** calculate metrics for each position over images that have a true digit at that position, then average positions. This prevents the common first digit from dominating, but has higher variance for rare positions.
2. **Sample macro-average:** calculate metrics per image over its valid digits, then average images. This gives every number equal weight.
3. **True-position conditional:** calculate the old valid-target-position metrics without the sentinel, explicitly labelled as a diagnostic. This is useful for comparison with earlier greedy tables but is not the primary required metric because it ignores extra digits.

The report must not present R2/NSE as sufficient OCR metrics. Add exact full-sequence accuracy, exact length accuracy, per-character accuracy over aligned valid positions, and normalized Levenshtein distance. These identify cases where numeric digit error is small but the whole number is still wrong. Use the same sentinel-aware per-image losses for the paired bootstrap.

## 6. Analysis experiments and report figures

The following figures are enough for a strong report; avoid producing decorative plots without a question.

1. **Dataset overview:** one week of `MT_235` load and the stable-client aggregate as a negative-control diagnostic, plus a seasonal profile by hour and weekday. Include the audit's exact row count, train-only stable-client count (expected 155) and whole-series audit count (154), structural-zero counts, missingness, and timestamp-regularity checks.
2. **Autocorrelation and baseline difficulty:** ACF at lags 1, 24, and 168; scatter of persistence/seasonal baseline error by hour of day. This motivates the recurrent context and shows whether the problem is nontrivial.
3. **Task 1 forecast panel:** representative test windows showing true `MT_235`, persistence, best linear baseline, attention-free GRU, and attentive BiGRU; include a zoomed two-step forecast.
4. **Task 1 attention ablation plot:** completed Phase 1 attention/no-attention mean ± standard deviation across exactly three seeds with paired intervals.
5. **Task 1 attention heatmaps:** temporal attention weights over the 168-hour input for several test cases, with markers at 24-hour and 168-hour lags and one failure case.
6. **Task 2 horizon plot:** MAE and RMSE at horizons 1–5 for the completed Phase 1 models, showing error accumulation.
7. **Task 2 attention examples:** encoder-time versus decoder-horizon attention heatmaps and representative failures.
8. **Task 3 qualitative grid:** separate Setting A and Setting B correct/incorrect SVHN predictions from I0, I1, and the strongest improvement, showing input, ground truth, decoder/beam, prediction, and attention overlay.
9. **Task 3 improvement breakdown:** exact-sequence accuracy by number length, character/length errors, sentinel-aware numeric metrics, DRAM/MBR/length-decoding intervals, and I2–I6 paired intervals.
10. **Compute/accuracy trade-off:** parameter count, training time, and test RMSE/sequence accuracy for the main models and comparators. This prevents a small improvement from being presented without cost.

Required tables: dataset audit statistics and splits; complete model hyperparameters and parameter counts; primary metrics with mean ± standard deviation over seeds; per-horizon Task 2 metrics; Task 3 numeric and OCR metrics; I0–I6 improvement rows with Holm-adjusted p-values and paired confidence intervals; and the final run/checkpoint manifest.

## 7. Execution order and coder TODO

Tasks 1–2 are paused with their Phase 1 and Phase 1b results final for now. The only new execution is the Task 3 improvement study. Every final run uses exactly seeds `0, 1, 2`.

### Phase 0: locked data and export setup

1. Download the UCI and SVHN files, record checksums, parse the audited decimal-comma UCI file, and verify the exact row/image counts.
2. Freeze `MT_235`, the target-only/seasonal-residual feature definitions, the chronological time-series splits, the SVHN train+extra validation hash split, both SVHN input caches/settings, and `MAX_LEN=5` after dropping the one six-digit training image.
3. Implement leakage-safe windowing, train-only scaling, inverse transforms, the four required metric functions, sentinel-aware sequence metrics, validation-only decoding-width selection, and hand-worked unit tests.
4. Implement checkpoint/resume with model, optimizer, scheduler, scaler, RNG, epoch, and data-order state. Validate resumption on the shared A100 before any final run.

Phase 0 is CPU/I/O work plus the three-seed extra-data caching/resume smoke test. It builds separate oracle-crop and whole-image caches, records the completed exact/near-duplicate audit and its manual-review decision, and exports `manifests/dataset_manifest.json`, `manifests/split_manifest.json`, `manifests/environment.json`, and `manifests/checksums.txt`.

### Phase 1: report-required baselines and course models only

Phase 1 is intentionally limited to the models needed for the template's experimental-results tables, but its preprocessing and recipe now follow the strongest public GitHub/Kaggle practice on these exact datasets:

- **Task 1:** completed target-only `MT_235` history, per-target scaling, timestamp-aligned daily/weekly bases, corrected sparse/full-window ridge, required seasonal-residual Attentive BiGRU-2, and the same residual GRU without attention.
- **Task 2:** the same target-only/residual preprocessing, corrected direct ridge, required seasonal-residual attentive LSTM encoder–decoder, and the same residual encoder–decoder without attention.
- **Task 3 Setting A:** GT union-box crop expanded 1.3x, square 64x64 resize, random train 54x54 crop, center eval crop, train+extra with deterministic 90/10 validation, required CNN+attentive GRU with length head, no-attention GRU, public-practice multi-head CNN, and CRNN-CTC diagnostic. Existing checkpoints are re-evaluated with validation-selected beam/MAP decoding; no retraining is needed for this setting.
- **Task 3 Setting B:** complete whole images resized directly to 64x64 with no boxes, the same random/center 54x54 crop protocol, a separate deterministic 90/10 train+extra split, and separately trained versions of the same four models. The old crop-trained-checkpoint whole-image number is retained only as a cross-setting diagnostic.

The Phase 1 tables are already complete. For deterministic CPU baselines, preserve one row per seed even when predictions are identical. Phase 1b is analysis of the completed linear-highway/ensemble runs and does not replace the registered models. No new Task 1–2 runs are authorized in this plan.

Required Phase 1 deliverables:

- `results/phase1/task1_metrics_by_seed.csv` and `results/phase1/task1_metrics_summary.csv` with R2, NSE, MAE, RMSE for every baseline/model and seed.
- `results/phase1/task2_metrics_by_seed.csv`, `task2_metrics_summary.csv`, and `task2_metrics_by_horizon.csv` with pooled and horizon-wise metrics.
- `results/phase1/task3_metrics_by_seed.csv` and `task3_metrics_summary.csv` with R2, NSE, MAE, RMSE, exact-string accuracy, character accuracy, length accuracy, and normalized edit distance separately for Setting A, Setting B, and the labelled cross-setting stress diagnostic. Include decoder names `beam_primary`, `mbr_edit_secondary`, `length_rescore_secondary`, `dram_fb_secondary`, `ctc_prefix_primary`, optional `ctc_seqmap_diagnostic`, `multihead_map_primary`, and `greedy_diagnostic`. The official GRU/CTC rows must use validation-selected beams; the multi-head rows must use exact MAP decoding.
- `results/phase1/task1_phase1b.csv` and `task2_phase1b.csv` with per-seed highway metrics, three-seed ensemble metrics, validation-selected ridge blending weights, confidence intervals, and the explicit conclusion that the observed point improvements over full-window ridge are not statistically significant.
- `results/phase1/predictions/` containing one machine-readable prediction file per model/seed and the per-origin/per-sample error files required for paired comparisons.
- `figures/phase1/` containing the dataset overview, residual-baseline forecast panels, Task 2 horizon plot, separate Setting A/Setting B SVHN qualitative/error plots, the two-setting comparison, and the crop-trained-checkpoint stress diagnostic.

### Task 3 improvement study: must-have order

1. **Lock the reference and decoder evidence:** preserve I0 primary beam rows and I1 DRAM rows in both settings; verify validation-selected widths/lambdas, per-image predictions, and paired bootstrap exports.
2. **Run single-factor I2 and I3** in both A and B, three seeds each. I2 is random-resized scale augmentation; I3 is label smoothing. Export validation curves and exact parameter counts.
3. **Run single-factor I4 and I5** in B, three seeds each. I4 is the identity-initialized affine STN; I5 is the 96/80 higher-resolution input. These directly target the B scale/localization gap.
4. **Run I6** in B only after I2–I5 checkpoints and validation tables are present. It is the fixed combined recipe, not a test-driven combination.
5. **Run paired image bootstrap, Holm correction, error taxonomy, and figures**. If a variant does not improve validation, still report its test result as a pre-registered negative result; do not retune on test.

Checkpoint every epoch and every 1,000 steps; store optimizer, scheduler, RNG, data-order, validation decoder selection, and test predictions. The new Task 3 allowance is the 9.75 active A100 hours in Section 4.4; reserve 15 wall-clock hours for preemption and resume.

### Report construction and required export contract

After the Task 3 improvement study, I will write the final report as a `.docx` that follows the course template exactly, not LaTeX. The template structure read from `Final Major Assignment Report 2026.docx` is:

- Part 1: Theoretical Background — RNNs, encoder–decoder/sequence-to-sequence models, and attention mechanisms.
- Part 2: Practical Implementation — Task 1 dataset description, problem description, architecture/design, experimental results; Task 2 problem description, architecture/design, experimental results; Task 3 dataset description, problem description, architecture/design, experimental results.
- Part 3: Conclusion.
- References.

Use the supplied `.docx` as the formatting starting point: preserve its cover/header/footer, course and faculty labels, heading numbering, section order, and native Word styles. Fill the existing placeholders and add only the requested tables/figures inside the existing sections; do not convert the report to LaTeX or invent a different chapter structure.

The coder must export the following exact facts so every template field can be filled without rereading raw data or guessing from plots.

| Export | Required contents |
|---|---|
| `dataset_manifest.json` | Dataset name, authoritative download URL, raw filename, checksum, raw observations, hourly observations, all 370 original attributes, train-only stable-client audit count (expected 155), frequency, time period, `pred_column=MT_235`, target mean/std/min/max/zero count, UCI decimal-comma/DST convention, SVHN train/extra/test counts, excluded six-digit count, Setting A GT-crop geometry, Setting B whole-image geometry with no boxes, resized canvas, sequence vocabulary, `MAX_LEN=5`, duplicate-audit result, and train/validation/test length histograms. |
| `split_manifest.json` | Exact train/validation/test timestamp boundaries and window counts for Task 1 and Task 2; exact SVHN train+extra train/validation/test image counts and sequence-length counts. Include the formulas `N-L-2+1` and `N-L-5+1` and the realized counts after boundary filtering. |
| `phase1/task1_model_specs.json` | Title, objective, input shape/window, output shape, architecture diagram data, parameter count, optimizer, learning rate, batch size, epochs, patience, loss, seed list, and checkpoint path for every Task 1 model. |
| `phase1/task2_model_specs.json` | The same fields for the encoder–decoder models, including teacher-forcing schedule, horizon embeddings, and attention configuration. |
| `task3_model_specs.json` | The same fields for the frozen main CNN-attention GRU, its I1 right-to-left decoders, and I2–I6 improvement variants in Settings A/B, including boxes, crop geometry, STN/high-resolution geometry, train+extra counts, token-map size, vocabulary, length head, augmentation, teacher forcing, validation-selected beam width, MBR/DRAM lambda values, checkpoint rule, and exact parameter count. |
| `*_metrics_by_seed.csv` | One row per model/seed with R2, NSE, MAE, RMSE and, for Task 3, setting, decoding mode/width, exact-string accuracy, character accuracy, length accuracy, normalized edit distance, and both sentinel-aware primary and true-position diagnostic metrics. |
| `task2_metrics_by_horizon.csv` | One row per model/seed/horizon 1–5 with R2, NSE, MAE, RMSE, plus pooled and macro summaries. |
| `improvement_summary.csv` | I0–I6 means, standard deviations, medians, per-seed values, setting, parameter counts, training hours, decoder width/lambda, and primary/secondary claim status. |
| `paired_error_differences.csv` | Same-seed per-origin Task 1/2 losses and same-seed per-image Task 3 losses for each Task 3 improvement versus I0; include model names, seed, image ID/origin, horizon where applicable, true value, prediction, and loss. |
| `dm_tests.csv` | DM statistic, HAC lag, loss type, raw p-value, Holm-adjusted p-value, contrast, task, and seed-averaging rule. |
| `bootstrap_confidence_intervals.csv` | 10,000-replicate block/image bootstrap intervals for RMSE, MAE, NSE, exact-string accuracy, and I1/I2/I4/I5 contrasts, with I3/I6 labelled secondary. |
| `references.csv` | Citation key, exact title, authors, year, venue/DOI or URL, and which report section uses it. |
| `figure_index.json` | Figure filename, caption, task, source CSV/JSON, and the claim the figure supports. |

The code must also export these final report figures as PNG and editable SVG where possible:

1. `figures/dataset_overview.png`: UCI target/covariate load overview, hourly frequency, period, and ACF at lags 1/24/168.
2. `figures/task1_baseline_forecasts.png`: `MT_235` test forecasts for persistence, timestamp-aligned seasonal, corrected ridge, residual no-attention GRU, and residual Attentive BiGRU.
3. `figures/task1_attention_ablation.png`: completed Phase 1 attention/no-attention RMSE/MAE with paired intervals.
4. `figures/task2_horizon_metrics.png`: Task 2 RMSE/MAE/R2/NSE at horizons 1–5 for completed Phase 1 models.
5. `figures/task2_attention_maps.png`: completed decoder attention maps and representative horizon errors.
6. `figures/task3_examples_attention.png`: Setting A and Setting B SVHN input, ground truth, beam/DRAM prediction, and per-character attention for correct and failed examples.
7. `figures/task3_length_errors.png`: Setting A versus Setting B exact-string accuracy by sequence length, character/length errors, oracle-crop versus trained whole-image results, cross-setting stress results, and improvement intervals.
8. `figures/task3_decoding_methods.png`: primary beam/CTC/MAP results and secondary MBR, length-rescoring, and DRAM forward-backward results, with all selection decisions made on validation.
9. `figures/compute_accuracy_tradeoff.png`: parameter count, active training time, and primary test metric.

### Phase 4: report review and revision timeline

The timeline is: finalize the existing Phase 1/Phase 1b exports; run the Task 3 improvement matrix and paired statistics; construct the `.docx` report in the exact template structure; then review the report using the available academic-writing/paper-review skills and revise it based on that review. The reviewed and revised `.docx`, not the first draft, is the handoff artifact.

**Operational stopping rule:** if the schedule becomes tight, finish the frozen Task 1–2 report exports, the Task 3 I0/I1 results, and the single-factor B geometry runs before I3/I6. The three-seed paired image bootstrap is the evidence standard.

## 8. References to cite in the eventual report

- Bahdanau, D., Cho, K., and Bengio, Y. (2015). *Neural machine translation by jointly learning to align and translate*.
- Goodfellow, I. J. et al. (2014). *Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks*.
- He, P. et al. (2019). *Deep Learning for Scene Text Recognition*.
- Kumar, S., and Byrne, W. (2004). *Minimum Bayes-Risk Decoding for Statistical Machine Translation*.
- Ba, J., Mnih, V., and Kavukcuoglu, K. (2015). *Multiple Object Recognition with Visual Attention*.
- Bautista, D. C., and Atienza, R. (2021). *Scene Text Recognition with Permuted Autoregressive Sequence Models*.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*.
- Jaderberg, M. et al. (2015). *Spatial Transformer Networks*.
- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*.
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*.
- Oreshkin, B. N. et al. (2020). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting*.
- Shi, B., Bai, X., and Yao, C. (2017). *An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition*.
- Shi, B. et al. (2018). *ASTER: An Attentional Scene Text Recognizer with Flexible Rectification*.
- Xu, K. et al. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*.
- Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?*

Use the paper landing pages/DOIs in the final bibliography, verify author lists and venue formatting before submission, and do not claim a paper introduced an exact implementation feature unless the paper supports that claim.
