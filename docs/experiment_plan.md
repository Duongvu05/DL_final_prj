# Group 12 experiment plan

## 0. Decision summary and scope

**Task 1–2 dataset:** [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), UCI Machine Learning Repository. Use the original 370-customer, 15-minute load table, aggregate each customer to hourly mean load, and set `pred_column = MT_235`. The primary Phase 1/2 models use the target's own history plus calendar and seasonal-residual features; they do not use the 155 other clients as neural inputs.

**Task 3 dataset:** [SVHN, Format 1: Full Numbers](https://ufldl.stanford.edu/housenumbers/), not the cropped-digit format. Each original image has a `digitStruct.mat` annotation containing the ordered digit bounding boxes and labels. The target is the variable-length digit string in left-to-right order.

This pairing is deliberately distinct from the registered datasets listed in the brief: it is neither ETTh1, Hanoi/Open-Meteo weather, UCI Bike Sharing, CCPD, UFPR-AMR, nor multi-digit handwritten MNIST. The time-series task is not treated as a random IID regression problem: the central experiment is whether recurrent memory, temporal attention, selective state-space memory, and cross-customer context improve forecasts over seasonal and linear lag baselines. The image task is not treated as digit classification: it uses the full scene image and predicts the entire ordered number.

The report should make two claims only if the results support them:

1. Attention improves access to relevant daily/weekly history or spatial digit features, rather than being assumed to help.
2. The required RNN families remain competitive with strong modern comparators under a leakage-safe, repeated-seed protocol.

The main models are intentionally implementable by one student team on an RTX 3060, with the A100 reserved for repeated final runs and the Phase 2 Mamba matrix.

## 1. Dataset choice

### 1.1 Task 1–2: primary dataset

**Dataset:** ElectricityLoadDiagrams20112014.

**Download and provenance.** The UCI landing page was checked on 2026-11-17 and is live; its Download control provides the archive. The data are 15-minute electricity consumption measurements for 370 customers from 2011–2014. The original SVHN and UCI pages are the authoritative links to put in the registration sheet, rather than a third-party mirror. The UCI page describes the file format and variables; preserve the original archive and record its SHA-256 after download.

**Audit-informed target and preprocessing for the experiment.**

The coder's audit found that `MT_001` is exactly zero for all of 2011 and only becomes live in 2012. That would make a chronological split deceptively easy and would violate the intent of a meaningful forecasting comparison. The target is therefore permanently `MT_235`: it belongs to the full-history, low-zero group and was among the audited hardest targets for persistence (`R²=0.27` for the audited last-15%-of-live-period test). This target choice is frozen before final model training; no test metric from the final experiment may be used to change it.

- Parse the semicolon-separated, decimal-comma table and its timestamp column. Convert the 15-minute measurements to hourly means, yielding exactly 35,064 hourly timestamps from 2011-01-01 through 2014-12-31 after the audit's row convention.
- Use `MT_235` as the fixed target. The main model input is target-only: the previous 168 hourly values of `MT_235`, target-hour/calendar features, and timestamp-aligned lag features at 24, 48, and 168 hours. The train-only stable-client audit returns 155 eligible clients (the whole-series audit returns 154), but this client set is exported for audit and may be used only as a documented negative-control experiment after the primary results; it is not a Phase 1/2 neural input.
- Add known-at-forecast-time features: `sin/cos(hour/24)`, `sin/cos(day_of_week/7)`, day-of-year sin/cos, and an explicit DST-transition flag. Do not use future observations as calendar features.
- Fit a per-target StandardScaler on the training interval only. For each target timestamp `τ=t+h`, define seasonal bases by timestamp-aligned lookup `b24(τ)=y(τ-24h)` and `b168(τ)=y(τ-168h)`, not by blindly subtracting rows from the forecast origin. Keep the UCI DST values as observed, but export DST flags and any unavailable seasonal lookup counts.
- Check timestamp regularity, duplicate timestamps, missing values, negative loads, constant columns, and zero-heavy columns before windowing. Short gaps of at most two hours may be linearly interpolated using adjacent observations; longer gaps must be removed from usable windows and counted. Never interpolate using validation or test values when constructing a training feature.
- Fit the feature standardization parameters on the training interval only. Inverse-transform predictions before all reported metrics.
- Main input window is `L = 168` hourly steps (one week). This matches common public practice while retaining the full daily/weekly context. Task 1 predicts `y[t+2]` exactly; Task 2 predicts `[y[t+1], ..., y[t+5]]`.

**Why this is a good deep-learning test.** The audited target has a large scale (`mean≈792` kW), full history, and substantially weaker persistence than the easiest customers. Public projects on this exact dataset show that daily seasonality, ridge/linear lag models, and residual learning over seasonal naive are stronger starting points than a large multi-client GRU. The Phase 1 base therefore tests the required RNN where it is methodologically plausible: target history plus calendar, with a linear/seasonal skip. The Mamba question is whether selective memory improves that matched residual architecture, not whether a high-capacity model can memorize 155 exogenous channels.

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
- Seeds: exactly `0, 1, 2` for every run; use the same three seeds for main models, baselines, Mamba models, ablations, and smoke/checkpoint tests. A smoke test may use fewer epochs, but it may not use fewer seeds and is not a final result.

The design is SOTA-informed but requirement-compliant. It uses the recurrent sequence encoder and attention pattern established for sequence-to-sequence learning, while the known calendar covariates and interpretable temporal weighting are informed by **“Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting” (Lim et al., 2021)**. Comparators include **“N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting” (Oreshkin et al., 2020)**, **“Are Transformers Effective for Time Series Forecasting?” (Zeng et al., 2023)**, and **“A Time Series is Worth 64 Words: Long-term Forecasting with Transformers” (Nie et al., 2023)**. The novel project contribution is a controlled, matched comparison of recurrent attention and selective state-space memory at the exact short horizon required by the course, with repeated seeds and formal paired tests.

**Phase 2 Mamba method: parameter-matched Seasonal-residual Mamba-2–GRU hybrid.** Replace only the BiGRU temporal encoder with two residual forward/reverse Mamba-2 blocks over the same target-only/lag/calendar input; retain the one-layer GRU readout, Bahdanau attention, `b24` seasonal skip, loss, and output head. Use `d_model=96`, `d_inner=64`, `d_state=16`, `d_conv=4`, `expand=2`, RMSNorm, and dropout 0.20; this gives approximately 175,181 parameters, within 5% of the 171,925-parameter Phase 1 attentive BiGRU. Match M1–M4 to M0 within 5% after the final parameter-count check. The GRU readout preserves the course RNN requirement.

This choice is grounded in **“Mamba: Linear-Time Sequence Modeling with Selective State Spaces” (Gu and Dao, 2023)** and the more efficient **“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality” (Dao and Gu, 2024)**. Its time-series adaptation is motivated by **“MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting” (Cai et al., 2024)** and **“TimeMachine: A Time Series is Worth 4 Mambas for Long-Term Forecasting” (Ahamed and Cheng, 2024)**. The project does not claim to reproduce MambaTS; it tests whether selective state updates help this shorter, multivariate electricity problem.

**Baselines and SOTA-informed comparators.** The locked execution set is the Phase 1/Phase 2 matrix in Section 4.4. The following literature comparators motivate the design but are not locked runs under the user's two-phase decision:

1. Persistence: `y[t+2] = y[t]`.
2. Timestamp-aligned seasonal naive: `y[t+2] = y[t+2-24h]` and `y[t+2] = y[t+2-168h]`.
3. Corrected sparse ridge on lags `{1, 2, 3, 24, 48, 168}` plus calendar features.
4. Full-window target-only ridge on all 168 target lags plus target-time calendar features; this is the linear diagnostic that reached about 68.7 kW RMSE.
5. Target-only GRU-2 without attention, with the same `b24` residual skip.
6. Phase 1 required Seasonal-residual Attentive BiGRU-2.
7. Phase 2 Seasonal-residual Mamba-2–GRU hybrid.
8. The 155-client GRU is a documented negative-control experiment only, not a headline baseline. N-BEATS, DLinear, and PatchTST remain literature context and are not locked runs.

The primary Mamba comparison is the hybrid versus a parameter-matched Attentive BiGRU with the same GRU readout, attention module, input, training budget, and seeds. The primary course comparison is the Attentive BiGRU versus the attention-free target-only GRU and best non-neural baseline. The SOTA comparators must not displace the required RNN experiments.

### 2.2 Task 2: five-step encoder–decoder forecast

**Question:** Does a target-only residual encoder–decoder with attention improve the five timestamp-aligned horizons over seasonal and linear baselines, and does a selective state-space encoder improve that matched residual architecture?

**Phase 1 required model: Seasonal-residual Bahdanau Attentive LSTM Encoder–Decoder.**

- Input: the same target-only 168-hour history, timestamp-aligned lag features, and calendar features as Task 1; output length exactly `H=5`.
- Seasonal skip: for each horizon `h`, compute `b24(t+h)=y(t+h-24h)` by timestamp lookup. The decoder predicts residuals `r_h=y(t+h)-b24(t+h)` and the final output is `b24+r_hat`.
- Encoder: two-layer LSTM, hidden size 64, dropout 0.20. Retain all encoder outputs and the final hidden/cell states.
- Decoder: one-layer LSTM, hidden size 64, initialized from the encoder final states. At each horizon, input the previous residual, known future calendar embedding, and a learned horizon embedding for `1..5`.
- Attention: additive attention over all encoder outputs at every decoder step. Report the five attention maps.
- Output: one residual scalar per decoder step. Train with teacher forcing ratio 1.0 in Phase 1 to match public sequence-to-sequence practice; at validation/test use fully autoregressive decoding. A scheduled-sampling sensitivity is deferred.
- Loss: mean MSE over the five standardized residuals, with equal horizon weights. Report all metrics after adding back the five seasonal bases and inverse transformation.
- Optimizer: AdamW, learning rate `5e-4`, weight decay `1e-4`, gradient clipping 1.0, batch size 128, maximum 60 epochs, patience 10, and exactly seeds `0, 1, 2`.

This model follows **“Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation” (Cho et al., 2014)** and **“Neural Machine Translation by Jointly Learning to Align and Translate” (Bahdanau et al., 2015)** for recurrent encoder–decoder and learned alignment, then adapts them to multi-horizon load forecasting. It is informed by **“Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting” (Lim et al., 2021)** because it conditions each future horizon on known covariates and analyzes temporal importance. The strongest modern comparators are N-BEATS (2020), DLinear (2023), and PatchTST (2023). Do not call the LSTM itself SOTA; call it an interpretable, requirement-compliant recurrent model tested against SOTA-informed and Mamba-based baselines.

**Phase 2 Mamba method: parameter-matched Seasonal-residual Mamba-2 encoder plus attentive LSTM decoder.** Replace only the two-layer LSTM encoder with two residual causal Mamba-2 blocks; retain the one-layer LSTM residual decoder, teacher forcing ratio 1.0, future-calendar inputs, horizon embeddings, timestamp-aligned `b24` skip, Bahdanau attention, loss, and output head exactly as in Phase 1. Use `d_model=112`, `d_inner=32`, `d_state=16`, `d_conv=4`, `expand=2`; this gives approximately 110,023 parameters, essentially equal to the 110,033-parameter Phase 1 attentive encoder–decoder. Match M1–M4 to M0 within 5% after the final parameter-count check. Initialize the LSTM decoder state from a learned projection of the final Mamba state. This remains an encoder–decoder network with the required LSTM decoder; no future target values enter the Mamba encoder.

The rationale is the same selective-memory lineage: Mamba (2023), Mamba-2/SSD (2024), MambaTS (2024), and TimeMachine (2024). MambaTS is particularly relevant because it studies variable-aware scans and selective dropout for multivariate long-horizon forecasting. Here the horizon is only five, so the experiment is a targeted test of whether the selective state update is useful even when a Transformer is not computationally necessary.

**Baselines and comparators.**

- Five-step persistence, timestamp-aligned daily seasonal, and weekly seasonal naive forecasts.
- Corrected sparse ridge and full-window target-only ridge direct multi-output regressors.
- LSTM residual encoder–decoder without decoder attention.
- Phase 1 required Seasonal-residual Attentive LSTM encoder–decoder.
- Phase 2 Seasonal-residual Mamba-2 encoder + attentive LSTM decoder.
- The old 155-client LSTM and N-BEATS/DLinear/PatchTST remain diagnostics/literature context, not headline runs.

Report both pooled metrics and metrics separately at horizons 1–5. The key failure mode to analyze is a model that has a good average but deteriorates sharply at horizon 5.

### 2.3 Task 3: CNN encoder plus recurrent attention decoder

**Question:** Can a CNN encoder plus an attentive GRU decoder recover ordered variable-length house numbers both with oracle localization and from complete images, and does attention improve over global pooling and CTC alternatives in each setting?

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

**Phase 2 Mamba method: parameter-matched CNN + bidirectional Vision-Mamba token mixer + attentive GRU decoder.** Run the same Mamba architecture separately in Setting A and Setting B, preserving each setting's input protocol, CNN, length head, decoder, beam-width selection, and validation checkpoint rule. Do not use the currently configured 7.95M-parameter Mamba model: it is 28% larger than the 6.20M Phase 1 CNN-attention-GRU. Reconfigure M0 to `d_model=128`, two bidirectional layers, `d_state=16`, `d_conv=4`, `expand=2`, and 2-D positions; this is approximately 6.20M parameters with the existing CNN/decoder. Match M1–M4 to M0 within 5% after the final parameter-count check. The GRU decoder and visual attention remain mandatory, so this is still a CNN encoder plus RNN decoder rather than a pure Mamba recognizer.

This is an OCR-specific adaptation of **“Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model” (Zhu et al., 2024)** and **“VMamba: Visual State Space Model” (Liu et al., 2024)**, which address the position sensitivity of visual tokens and bidirectional/2-D selective scanning. The original **“Mamba: Linear-Time Sequence Modeling with Selective State Spaces” (Gu and Dao, 2023)** supplies the selective state-space mechanism. The project novelty is a controlled test on both the crop/extra protocol and the harder no-localization protocol, while retaining the required recurrent decoder.

**Baselines and comparators.**

1. In each setting, Phase 1 CNN + attentive GRU with the auxiliary length head, reported with validation-selected beam search.
2. In each setting, the same CNN + GRU decoder without visual attention, also reported with validation-selected beam search.
3. In each setting, the public-practice multi-head CNN length/position classifier with exact MAP decoding.
4. In each setting, the CRNN-CTC diagnostic baseline with validation-selected prefix beam search; report its protocol caveat honestly.
5. In each setting, Phase 2 CNN + parameter-matched bidirectional Vision-Mamba token mixer + attentive GRU decoder.
6. The old crop-trained-checkpoint-on-whole-image evaluation remains a labelled cross-setting stress diagnostic, not Setting B. PARSeq remains literature context and is not a locked run.

Primary image metrics are computed on the decoded sequence, not on teacher-forced logits. Also report exact full-string accuracy, character accuracy, normalized Levenshtein distance, and length accuracy; these are necessary because four numeric regression metrics do not fully describe OCR quality.

### 2.4 Phase 1 results and the Phase 1b linear-highway analysis

The completed three-seed Phase 1 results establish the honest baseline before Mamba. Task 1 test RMSE is persistence `199.5` kW, daily seasonal `94.1`, weekly seasonal `105.9`, sparse ridge `79.1`, full-window ridge `68.78`, residual BiGRU `70.02±0.35`, and attentive residual BiGRU `70.68±0.76` (`R²≈0.909`). Task 2 pooled RMSE is daily seasonal `94.1`, sparse ridge `79.1`, full-window ridge `70.76`, residual LSTM encoder–decoder `73.45±1.38`, and attentive residual encoder–decoder `73.66±0.85`. Deep models beat full-window ridge on validation but not on the held-out test regime, so the report must not claim that attention or recurrence beats ridge. The Task 1 deep-versus-daily DM result is significant (`p=0.007`), while deep-versus-full-window-ridge and attention-versus-no-attention are not significant.

The user-requested full-window linear highway is a documented **Phase 1b supplementary improvement**, not a replacement for the registered Phase 1 models and not a new Mamba claim. It is the LSTNet-style jointly trained linear skip over the standardized target window and target-hour features. The attentive highway model gives Task 1 `69.30±0.44` kW across seeds; averaging the three seed forecasts gives `68.49`, and validation-selected blending with full-window ridge (`0.9` highway ensemble + `0.1` ridge) gives `67.60`. For Task 2, the corresponding values are `72.37±1.12`, `70.60`, and `69.51` with validation-selected weights `0.8/0.2`. The Task 1 blend versus full-window ridge has bootstrap difference `-1.18` kW with 95% CI `[-6.02,+3.96]` and DM `p≈0.6`; therefore these are promising point estimates, not statistically established wins. Report them in a separate Phase 1b table and use the full-window ridge as the primary non-neural reference. Phase 2 Mamba is evaluated against both that ridge reference and the matched recurrent control; if it also fails the pre-registered paired test, report that result rather than tuning until it wins.

The previously exported crop-setting Task 3 results (`95.76%±0.18` attention GRU, `95.59%` pooled GRU, `95.09%` multi-head CNN, `95.99%±0.14` CRNN-CTC) are **greedy-decoding provisional values**. They must be replaced in the report tables by validation-selected beam/MAP results under Setting A. The current `21.9%` whole-image result (roughly `19–24%` across models/seeds) is only the cross-setting stress diagnostic; it must not be labelled as the trained whole-image result. Setting B requires the new whole-image training runs specified in Section 4.4.

## 3. Ablation matrix

Every ablation below uses the same split, preprocessing, optimizer search budget, early-stopping rule, and checkpoint policy as its parent model. Mamba claims are evaluated with same-seed paired predictions: seed `s` for a Mamba model uses the same data split, batch order, augmentation stream, and early-stopping budget as seed `s` for its matched control. Widths are tuned before the final run so trainable parameter counts differ by at most 5%; report the exact counts. Exactly seeds `0, 1, 2` are used even for smoke tests; a smoke checkpoint is never evidence for a final Mamba claim.

### Task 1 ablations

| ID | Variant | What it tests |
|---|---|---|
| T1-P0 | Phase 1 seasonal-residual Attentive BiGRU-2, target-only input | Required course model on the public-practice base |
| T1-P1 | Same residual model without temporal attention | Required no-attention counterpart |
| T1-P2 | Same model without the `b24` residual skip | Diagnostic: value of residual learning; not a Phase 1 gate |
| T1-P3 | Same model with 155-client panel input | Negative control for the observed overfit; not a headline model |
| T1-M0 | Seasonal-residual Mamba-2–GRU hybrid with attention | Proposed Mamba method |
| T1-M1 | Parameter-matched Phase 1 residual BiGRU with the same GRU readout and attention | Matched non-Mamba control |
| T1-M2 | Mamba static-state control: replace input-dependent `Δ/B/C` selection with learned time-invariant state parameters, retaining equal-width capacity | Whether selective state updates matter |
| T1-M3 | Mamba without depthwise local convolution (`d_conv=1`) with compensating projection width | Whether the local convolution sub-part matters |
| T1-M4 | Forward-only Mamba scan instead of forward+reverse scan | Whether bidirectional state aggregation matters |

The decisive Task 1 Mamba evidence is T1-M0 versus T1-M1 and T1-M2. T1-M3/T1-M4 explain which Mamba sub-parts drive any gain; they are not allowed to be omitted if the project claims that selective state memory, rather than just extra parameters, caused the improvement.

### Task 2 ablations

| ID | Variant | What it tests |
|---|---|---|
| T2-P0 | Phase 1 seasonal-residual attentive LSTM encoder–decoder | Required course model on the public-practice base |
| T2-P1 | Same residual encoder–decoder without decoder attention | Required no-attention counterpart |
| T2-P2 | Same model without the `b24` residual skip | Diagnostic: value of residual learning |
| T2-P3 | Same model with 155-client panel input | Negative control for the observed overfit |
| T2-M0 | Seasonal-residual Mamba-2 encoder + attentive LSTM decoder | Proposed Mamba method |
| T2-M1 | Parameter-matched Phase 1 residual LSTM encoder + identical decoder | Matched non-Mamba control |
| T2-M2 | Mamba static-state control with input-independent `Δ/B/C` | Whether selective updates matter |
| T2-M3 | Mamba encoder without depthwise local convolution with matched projection capacity | Local convolution contribution |
| T2-M4 | Mamba encoder with decoder attention removed | Interaction between Mamba memory and alignment |

The decisive Task 2 Mamba evidence is T2-M0 versus T2-M1 and T2-M2, reported both pooled and at each horizon. T2-M3 and T2-M4 are mechanism checks, not optional tuning variants.

### Task 3 ablations

| ID | Variant | What it tests |
|---|---|---|
| T3-P0 | Setting A or B CNN + visual-attention GRU + length head | Required course model in each input setting |
| T3-P1 | Same setting-specific model without visual attention | Required no-attention counterpart |
| T3-P2 | Setting-specific multi-head CNN length/position classifier | Common public-practice non-recurrent baseline |
| T3-P3 | Setting-specific CRNN-CTC | CTC diagnostic with prefix-beam decoding |
| T3-P4 | Crop-trained checkpoint evaluated on whole images | Cross-setting localization/scale diagnostic only |
| T3-M0 | Setting A or B CNN + bidirectional Vision-Mamba token mixer + attentive GRU decoder | Proposed Mamba method in both settings |
| T3-M1 | Setting-specific CNN + parameter-matched bidirectional GRU token mixer + identical attentive GRU decoder | Matched non-Mamba visual control |
| T3-M2 | Setting-specific Vision-Mamba static-state control with input-independent selection parameters | Whether selective visual state updates matter |
| T3-M3 | Setting-specific forward-only visual Mamba scan | Whether bidirectional visual scanning matters |
| T3-M4 | Setting-specific Mamba without learned 2-D position embeddings | Whether positional marking is essential for ordered digits |

The decisive Task 3 Mamba evidence is T3-M0 versus T3-M1 and T3-M2 separately in Setting A and Setting B, using validation-selected beam exact-string accuracy and the required numeric metrics. T3-M3/T3-M4 isolate the two visual adaptations motivated by Vision Mamba: bidirectional context and explicit position marking. A Mamba gain in only the oracle setting must not be generalized to end-to-end whole-image recognition.

### 3.1 What counts as evidence that Mamba works

For each task, pre-register the Mamba claim as: the Mamba method must have lower mean test RMSE than its matched control, and the 95% paired bootstrap interval for the primary loss difference must exclude zero in the favorable direction. Every contrast uses exactly seeds `0, 1, 2`; if the interval includes zero, write “no reliable Mamba improvement” rather than selecting a favorable seed. Use the same criterion for the Mamba-vs-static-state comparison. This is stronger than comparing a single best run and directly tests the selective component.

The bootstrap resamples test origins in blocks for Tasks 1–2 and test images for Task 3. Also report the three paired seed-level differences and their mean/std, but do not use a t-test across seeds: `n=3` is too small for that claim. Mamba ablations must use the same model budget and parameter-count tolerance, not a larger hidden size or more epochs.

## 4. Evaluation protocol and leakage controls

### 4.1 Splits

**Time series:** chronological 70% train, 15% validation, 15% test by hourly timestamp. Construct a window only when the forecast target lies wholly in its split. It is acceptable for a validation/test input window to use historical context immediately before the split boundary, because that is available at prediction time; it is not acceptable to use any future value. Fit scalers, feature selection, missing-value rules, and all hyperparameters using train/validation only. Test is run once after model selection.

**SVHN:** remove the single six-digit train example, merge the remaining official train and extra images, and split the resulting 235,754-image pool 90/10 into train/validation with a fixed hash manifest stratified by sequence length. Keep the official 13,068-image test archive untouched. Train and evaluate two separate settings: Setting A crops every image with its own GT union box expanded 1.3x and is labelled oracle localization; Setting B uses the complete image resized to `64x64` with no boxes at train or test time. The old crop-checkpoint-on-whole-image result is only a stress diagnostic. Any duplicate-image audit must happen before splitting; the completed audit found no exact SHA-1 duplicates, but found near-duplicate groups requiring manual review, including one train/test group. Resolve or explicitly report any confirmed cross-split near-duplicate before final metrics.

### 4.2 Reproducibility and preemption

- Final seeds are locked by the matrix in Section 4.4: exactly seeds `0, 1, 2` for every model and ablation. Report mean, standard deviation, median, and every seed; rank models by mean test RMSE or exact-sequence accuracy, not by the best seed.
- Set Python, NumPy, PyTorch, and data-loader seeds; use deterministic cuDNN where practical; record GPU type, software versions, commit hash, dataset checksum, and exact command line.
- Save the best validation checkpoint, optimizer state, scheduler state, scaler, RNG states, current epoch, test predictions, per-origin/per-horizon errors, attention weights, and split manifest. Never select a checkpoint using test metrics.
- Standardize on the training interval only, and compute all metric values after inverse transformation. Store both scaled loss and physical-unit metrics.

The RunAI pod is preemptible and only about 23 GB of the 40 GB A100 is available because another process holds approximately 17 GB. Use mixed precision, cap the batch size to fit under the free-memory limit, checkpoint at the end of every epoch and every 1,000 optimizer steps, and resume rather than restart after eviction. Install `mamba-ssm` against the existing PyTorch 2.8/CUDA 12.9 environment; if compilation fails, use the pure-PyTorch `mambapy` implementation with the same dimensions and record the backend. Do not compare backends as if they were methods.

### 4.3 Statistical tests

For time series, test each Mamba method against its parameter-matched recurrent control and its static-state control, in addition to the best non-neural baseline and attention-free recurrent baseline.

- Use the Diebold–Mariano test on paired per-origin losses. For Task 1 use squared error and absolute error separately; for Task 2 use the average five-horizon squared error per origin and additionally test each horizon. Because rolling origins overlap, use a Newey–West/HAC variance with truncation lag `H-1` (or a conservative 24-hour lag for hourly forecasts) and report two-sided p-values with a 5% threshold.
- Use a paired moving-block bootstrap with 10,000 replicates over chronological test origins, block length 168 hours. Report the 95% confidence interval for the difference in RMSE, MAE, and pooled NSE. Do not resample individual overlapping rows independently.
- For a main Mamba contrast, average the same-seed loss difference at each test origin across seeds `0, 1, 2`, then run DM and the block bootstrap on that paired series. Also report each seed's result. For mechanism ablations, use the same procedure over the three paired predictions. Apply Holm correction within each task to the two primary Mamba hypotheses (`M0-M1`, `M0-M2`); label `M3/M4` as secondary mechanism tests.

For SVHN, use a paired bootstrap over test images with 10,000 replicates for the difference in exact-string accuracy, character accuracy, MAE, and RMSE, separately for Setting A and Setting B. Average the same-seed per-image loss differences before the bootstrap for M0-M1 and M0-M2. For exact-string correctness, add McNemar's test between the Mamba method and matched control and between the course main and strongest baseline. Apply Holm correction within each setting to the two primary Mamba hypotheses. Use two-sided tests and state that statistical significance is secondary to effect size and confidence intervals.

With exactly three seeds, the primary image test is the paired bootstrap over per-image differences after averaging the three same-seed correctness/loss values. Do not pretend that the 3 x 13,068 seed-image pairs are independent. Decode-width and rescoring selection is performed only on validation: GRU and CTC prefix beams test widths `{3,5,10}`, secondary lambdas use the pre-registered grid, mean validation exact match is pooled across the three seeds, ties choose the smaller width/lambda, and the chosen setting is frozen before test. Multi-head MAP decoding has no tunable width. McNemar's test may be reported separately for each seed as a descriptive secondary check, but the claim-supporting interval is the clustered per-image bootstrap; no seed-level t-test is allowed.

Do not claim that a p-value proves one deep architecture is universally superior. The valid claim is conditional on this dataset, horizon, preprocessing, and model budget.

### 4.4 Locked final run matrix and A100 budget

This is the final matrix, not a menu. A short smoke/checkpoint run is allowed before it, but it must also use exactly seeds `0, 1, 2` and cannot replace the listed runs. Every listed model/baseline evaluation uses exactly seeds `0, 1, 2`. Phase 1 must finish and export its tables before any Phase 2 run starts. The modern N-BEATS/DLinear/PatchTST papers remain literature context, but those extra models are not in this locked execution matrix because the user explicitly ordered Phase 1 to contain only report-required baselines/course models and Phase 2 to contain only Mamba proposals/ablations.

| Phase | Task/setting | Model family | Seeds | Runs | Estimated active A100 hours |
|---|---:|---|---:|---:|---:|
| 1 | 1 | Persistence, daily/weekly timestamp-aligned seasonal naive, sparse ridge, full-window target-only ridge | 0,1,2 | 12 model/seed evaluations | 0.0 |
| 1 | 1 | Seasonal-residual Attentive BiGRU-2 and same residual GRU without attention | 3 each | 6 | 0.8 |
| 1 | 2 | Persistence, daily/weekly timestamp-aligned seasonal naive, sparse ridge, full-window target-only ridge | 0,1,2 | 12 model/seed evaluations | 0.0 |
| 1 | 2 | Seasonal-residual attentive LSTM encoder–decoder and same residual decoder without attention | 3 each | 6 | 0.9 |
| 1 | 3A oracle crop | Original train+extra CNN-attention GRU, no-attention GRU, multi-head CNN, and CRNN-CTC training | 3 each | 12 model/seed evaluations, completed Phase 1 run | 15.6 already spent |
| 1 | 3A oracle crop | Existing cropped train+extra CNN-attention GRU, no-attention GRU, multi-head CNN, and CRNN-CTC checkpoints; re-evaluate with validation-selected beam/MAP decoding | 3 each | 12 decoding evaluations, no retraining | 0.5 |
| 1 | 3B whole image | Separately trained whole-image train+extra CNN-attention GRU, no-attention GRU, multi-head CNN, and CRNN-CTC | 3 each | 12 model/seed evaluations | 15.6 |
| 1 | 3A + 3B | Right-to-left attentive-GRU decoder for DRAM-style rescoring of the main attention GRU, one model per setting | 3 each | 6 additional model/seed evaluations | 2.0 |
| 1 | 3 diagnostic | Crop-trained checkpoint evaluated on whole-image inputs | 3 | 3 cross-setting evaluations | 0.3 |
| 1 | all | Three-seed smoke runs, extra-cache/checkpoint validation, and Phase 1 export verification | 0,1,2 | operational checks, not model/seed totals | 1.5 |
| 2 | 1 | Seasonal-residual Mamba-2–GRU T1-M0 and matched residual BiGRU T1-M1 | 3 each | 6 | 2.4 |
| 2 | 1 | Static-state, no-local-conv, forward-only T1-M2–M4 | 3 each | 9 | 2.7 |
| 2 | 2 | Seasonal-residual Mamba-2 encoder + attentive LSTM decoder T2-M0 and matched residual LSTM T2-M1 | 3 each | 6 | 3.0 |
| 2 | 2 | Static-state, no-local-conv, no-decoder-attention T2-M2–M4 | 3 each | 9 | 3.6 |
| 2 | 3A oracle crop | Parameter-matched CNN + Vision-Mamba token mixer T3-M0, matched visual-GRU T3-M1, and T3-M2–M4 ablations | 3 each | 15 model/seed evaluations | 18.3 |
| 2 | 3B whole image | The same parameter-matched Mamba/control/ablation matrix trained and evaluated without boxes | 3 each | 15 model/seed evaluations | 18.3 |
| 2 | all | Three-seed smoke runs, checkpoint validation, and Phase 2 export verification | 0,1,2 | operational checks, not model/seed totals | 1.5 |
| **Total** |  |  |  | **129 trained model/seed evaluations + 12 decoding-only re-evaluations** | **87.0 active hours** |

CPU-only seasonal/ridge metrics, Phase 1b blending, statistical tests, and report exports do not consume meaningful A100 time. Reserve **110 A100 wall-clock hours**: approximately 87 active training/evaluation hours plus 23 hours for preemption, resume, compilation, validation-width selection, right-to-left decoder training, and queue delay. Do not add PARSeq, N-BEATS, DLinear, PatchTST, extra hyperparameter searches, or non-Mamba ablations before the report is complete. The must-have scientific result is the two-setting Phase 1 table followed by the two-setting, parameter-matched Phase 2 Mamba/control/static-state matrix with seeds `0, 1, 2`.

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
3. **Task 1 forecast panel:** representative test windows showing true `MT_235`, persistence, best linear baseline, attention-free GRU, attentive BiGRU, and Mamba-2–GRU; include a zoomed two-step forecast.
4. **Task 1 Mamba ablation plot:** mean ± standard deviation across exactly three seeds for M0/M1/M2/M3/M4, with paired-bootstrap intervals for Mamba versus matched BiGRU and static-state controls.
5. **Task 1 attention/state heatmaps:** temporal attention weights and Mamba selection/state diagnostics over the 168-hour input for several test cases, with markers at 24-hour and 168-hour lags. Include a failure case, not only a success case.
6. **Task 2 horizon plot:** MAE and RMSE at horizons 1–5 for all key models, including Mamba encoder–decoder and matched LSTM. This is the main evidence for error accumulation and whether selective memory helps at later steps.
7. **Task 2 attention/state examples:** encoder-time versus decoder-horizon attention heatmaps, plus Mamba-vs-static-state diagnostics; discuss whether different horizons use different historical positions.
8. **Task 3 qualitative grid:** separate Setting A and Setting B correct/incorrect SVHN predictions from course-main, Mamba, and matched visual-GRU models, showing input image, ground-truth string, prediction, decoder type/beam width, and attention overlay for each decoded character.
9. **Task 3 setting and Mamba breakdown:** exact-sequence accuracy by number length for both settings, character accuracy by position, sentinel-aware numeric metrics, M0/M1/M2 confidence intervals, and separate early-EOS, wrong-digit, extra-digit, localization, and position-embedding failures.
10. **Compute/accuracy trade-off:** parameter count, training time, and test RMSE/sequence accuracy for the main models and comparators. This prevents a small improvement from being presented without cost.

Required tables: dataset audit statistics and splits; complete model hyperparameters and parameter counts; primary metrics with mean ± standard deviation over seeds; per-horizon Task 2 metrics; Task 3 numeric and OCR metrics; Mamba ablations with adjusted p-values and confidence intervals; and the final run/checkpoint manifest.

## 7. Two-phase execution order and coder TODO

The coder must not start Phase 2 until every Phase 1 run has finished, its checkpoints and predictions are saved, and its report tables have been exported. Every final run in both phases uses exactly seeds `0, 1, 2`.

### Phase 0: locked data and export setup

1. Download the UCI and SVHN files, record checksums, parse the audited decimal-comma UCI file, and verify the exact row/image counts.
2. Freeze `MT_235`, the target-only/seasonal-residual feature definitions, the chronological time-series splits, the SVHN train+extra validation hash split, both SVHN input caches/settings, and `MAX_LEN=5` after dropping the one six-digit training image.
3. Implement leakage-safe windowing, train-only scaling, inverse transforms, the four required metric functions, sentinel-aware sequence metrics, validation-only decoding-width selection, and hand-worked unit tests.
4. Implement checkpoint/resume with model, optimizer, scheduler, scaler, RNG, epoch, and data-order state. Validate resumption on the shared A100 before any final run.

Phase 0 is CPU/I/O work plus the three-seed extra-data caching/resume smoke test already budgeted in the Phase 1 “all” row of Section 4.4. It builds separate oracle-crop and whole-image caches, records the completed exact/near-duplicate audit and its manual-review decision, and exports `manifests/dataset_manifest.json`, `manifests/split_manifest.json`, `manifests/environment.json`, and `manifests/checksums.txt`.

### Phase 1: report-required baselines and course models only

Phase 1 is intentionally limited to the models needed for the template's experimental-results tables, but its preprocessing and recipe now follow the strongest public GitHub/Kaggle practice on these exact datasets:

- **Task 1:** target-only `MT_235` history, per-target scaling, timestamp-aligned daily/weekly bases, corrected sparse/full-window ridge, required seasonal-residual Attentive BiGRU-2, and the same residual GRU without attention.
- **Task 2:** the same target-only/residual preprocessing, corrected direct ridge, required seasonal-residual attentive LSTM encoder–decoder, and the same residual encoder–decoder without attention.
- **Task 3 Setting A:** GT union-box crop expanded 1.3x, square 64x64 resize, random train 54x54 crop, center eval crop, train+extra with deterministic 90/10 validation, required CNN+attentive GRU with length head, no-attention GRU, public-practice multi-head CNN, and CRNN-CTC diagnostic. Existing checkpoints are re-evaluated with validation-selected beam/MAP decoding; no retraining is needed for this setting.
- **Task 3 Setting B:** complete whole images resized directly to 64x64 with no boxes, the same random/center 54x54 crop protocol, a separate deterministic 90/10 train+extra split, and separately trained versions of the same four models. The old crop-trained-checkpoint whole-image number is retained only as a cross-setting diagnostic.

Run every Phase 1 model or baseline under exactly seeds `0, 1, 2`, export all Phase 1 tables, and stop. For deterministic CPU baselines, execute the same three-seed evaluation jobs and write one row per seed even when predictions are identical. Do not run Mamba, Mamba ablations, N-BEATS, DLinear, PatchTST, PARSeq, random-CNN controls, or exploratory non-Mamba ablations in Phase 1. Phase 1b is analysis of the already completed linear-highway/ensemble runs and does not replace the locked Phase 1 models. The expected Phase 1 A100 allowance is approximately 37.2 active hours, including whole-image retraining, oracle-crop decoding re-evaluation, six right-to-left decoder runs, extra-cache, and resume validation; CPU baselines, Phase 1b statistics, and blending are outside that estimate.

Required Phase 1 deliverables:

- `results/phase1/task1_metrics_by_seed.csv` and `results/phase1/task1_metrics_summary.csv` with R2, NSE, MAE, RMSE for every baseline/model and seed.
- `results/phase1/task2_metrics_by_seed.csv`, `task2_metrics_summary.csv`, and `task2_metrics_by_horizon.csv` with pooled and horizon-wise metrics.
- `results/phase1/task3_metrics_by_seed.csv` and `task3_metrics_summary.csv` with R2, NSE, MAE, RMSE, exact-string accuracy, character accuracy, length accuracy, and normalized edit distance separately for Setting A, Setting B, and the labelled cross-setting stress diagnostic. Include decoder names `beam_primary`, `mbr_edit_secondary`, `length_rescore_secondary`, `dram_fb_secondary`, `ctc_prefix_primary`, optional `ctc_seqmap_diagnostic`, `multihead_map_primary`, and `greedy_diagnostic`. The official GRU/CTC rows must use validation-selected beams; the multi-head rows must use exact MAP decoding.
- `results/phase1/task1_phase1b.csv` and `task2_phase1b.csv` with per-seed highway metrics, three-seed ensemble metrics, validation-selected ridge blending weights, confidence intervals, and the explicit conclusion that the observed point improvements over full-window ridge are not statistically significant.
- `results/phase1/predictions/` containing one machine-readable prediction file per model/seed and the per-origin/per-sample error files required for paired comparisons.
- `figures/phase1/` containing the dataset overview, residual-baseline forecast panels, Task 2 horizon plot, separate Setting A/Setting B SVHN qualitative/error plots, the two-setting comparison, and the crop-trained-checkpoint stress diagnostic.

### Phase 2: Mamba proposals and Mamba ablations only

Start Phase 2 only after the Phase 1 export checklist passes. Run exactly the M0/M1/M2/M3/M4 models in Section 4.4 with seeds `0, 1, 2`, reusing the Phase 1 data protocol and matched parameter budgets. Phase 2 Task 3 is run in both image settings so a Mamba result cannot be attributed only to oracle localization:

- Task 1: seasonal-residual Mamba-2–GRU hybrid, matched Phase 1 residual BiGRU, static-state Mamba, no-local-convolution Mamba, and forward-only Mamba.
- Task 2: seasonal-residual Mamba-2 encoder plus attentive LSTM decoder, matched Phase 1 residual LSTM encoder, static-state Mamba, no-local-convolution Mamba, and no-decoder-attention Mamba.
- Task 3 Setting A: oracle-crop train+extra CNN plus parameter-matched Vision-Mamba token mixer and attentive GRU decoder, matched visual-GRU mixer, static-state visual Mamba, forward-only visual Mamba, and no-position visual Mamba.
- Task 3 Setting B: the identical parameter-matched matrix trained and evaluated on whole images without boxes.

Export the same Phase 1 metrics plus `results/phase2/mamba_ablation_metrics_by_seed.csv`, `mamba_ablation_summary.csv`, `paired_error_differences.csv`, `dm_tests.csv`, `bootstrap_confidence_intervals.csv`, and `holm_adjusted_tests.csv`, with a `setting` column for Task 3. Export Mamba selection/state diagnostics and attention maps for both Task 3 settings in `figures/phase2/`. Record the pure-PyTorch backend and ONNX export status; do not treat ONNX export as a model comparison.

The locked Phase 2 allowance is approximately 49.8 active A100 hours. The total locked experiment budget is therefore approximately 87 active A100 hours. Reserve 110 A100 wall-clock hours as specified in Section 4.4. Use the verified pure-PyTorch implementation for the final runs; no CUDA `mamba-ssm` kernel is required for this matrix. Use the project virtual environment/`uv`, persistent storage, plain `nohup` for background jobs, and the pod keepalive procedure. Do not use `pm2`.

### Phase 3: report construction and required export contract

After Phase 2, I will write the final report as a `.docx` that follows the course template exactly, not LaTeX. The template structure read from `Final Major Assignment Report 2026.docx` is:

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
| `phase1/task3_model_specs.json` | The same fields for Setting A and Setting B CNN-GRU, no-attention GRU, multi-head CNN, CRNN-CTC, and the six right-to-left attentive-GRU runs, including whether boxes were used, GT union-box margin when applicable, 64x64/54x54 geometry, train+extra counts, CNN feature-map size, decoder vocabulary, length head, augmentation, teacher forcing, validation-selected beam/prefix-beam width, MBR/length/DRAM lambda values, MAP rule, checkpoint selection rule, and parameter count. |
| `*_metrics_by_seed.csv` | One row per model/seed with R2, NSE, MAE, RMSE and, for Task 3, setting, decoding mode/width, exact-string accuracy, character accuracy, length accuracy, normalized edit distance, and both sentinel-aware primary and true-position diagnostic metrics. |
| `task2_metrics_by_horizon.csv` | One row per model/seed/horizon 1–5 with R2, NSE, MAE, RMSE, plus pooled and macro summaries. |
| `mamba_ablation_summary.csv` | M0/M1/M2/M3/M4 means, standard deviations, medians, per-seed values, setting, parameter counts, training hours, Phase 1 base protocol, decoder width, and the pre-registered claim status. |
| `paired_error_differences.csv` | Same-seed per-origin Task 1/2 losses and same-seed per-image Task 3 losses for M0-M1 and M0-M2; include model names, seed, origin/image ID, horizon, true value, prediction, and loss. |
| `dm_tests.csv` | DM statistic, HAC lag, loss type, raw p-value, Holm-adjusted p-value, contrast, task, and seed-averaging rule. |
| `bootstrap_confidence_intervals.csv` | 10,000-replicate block/image bootstrap intervals for RMSE, MAE, NSE, exact-string accuracy, and the Mamba contrasts. |
| `references.csv` | Citation key, exact title, authors, year, venue/DOI or URL, and which report section uses it. |
| `figure_index.json` | Figure filename, caption, task, source CSV/JSON, and the claim the figure supports. |

The code must also export these final report figures as PNG and editable SVG where possible:

1. `figures/dataset_overview.png`: UCI target/covariate load overview, hourly frequency, period, and ACF at lags 1/24/168.
2. `figures/task1_baseline_forecasts.png`: `MT_235` test forecasts for persistence, timestamp-aligned seasonal, corrected ridge, residual no-attention GRU, and residual Attentive BiGRU.
3. `figures/task1_mamba_ablation.png`: M0/M1/M2/M3/M4 RMSE/MAE with 95% paired intervals.
4. `figures/task2_horizon_metrics.png`: Task 2 RMSE/MAE/R2/NSE at horizons 1–5 for Phase 1 and Mamba models.
5. `figures/task2_attention_state_maps.png`: decoder attention and Mamba state/selection diagnostics.
6. `figures/task3_examples_attention.png`: Setting A and Setting B SVHN input, ground truth, beam/MAP prediction, and per-character attention for correct and failed examples.
7. `figures/task3_length_errors.png`: Setting A versus Setting B exact-string accuracy by sequence length, character/length errors, oracle-crop versus trained whole-image results, cross-setting stress results, decoder-method comparison, and Mamba ablation intervals.
8. `figures/task3_decoding_methods.png`: primary beam/CTC/MAP results and secondary MBR, length-rescoring, and DRAM forward-backward results, with all selection decisions made on validation.
9. `figures/compute_accuracy_tradeoff.png`: parameter count, active training time, and primary test metric.

### Phase 4: report review and revision timeline

The timeline is: Phase 0 setup; Phase 1 required models and exported tables; Phase 2 Mamba models/ablations and paired statistics; Phase 3 `.docx` report construction in the exact template structure; then a review pass using the available academic-writing/paper-review skills, followed by revision of the report based on that review. The reviewed and revised `.docx`, not the first draft, is the handoff artifact.

**Operational stopping rule:** if the schedule becomes tight, finish Phase 1 exports and the Phase 2 matched Mamba/control/static-state matrix before adding any literature-only comparator or extra figure. The three-seed paired error analysis is the evidence standard.

## 8. References to cite in the eventual report

- Bahdanau, D., Cho, K., and Bengio, Y. (2015). *Neural machine translation by jointly learning to align and translate*.
- Goodfellow, I. J. et al. (2014). *Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks*.
- He, P. et al. (2019). *Deep Learning for Scene Text Recognition*.
- Kumar, S., and Byrne, W. (2004). *Minimum Bayes-Risk Decoding for Statistical Machine Translation*.
- Ba, J., Mnih, V., and Kavukcuoglu, K. (2015). *Multiple Object Recognition with Visual Attention*.
- Bautista, D. C., and Atienza, R. (2021). *Scene Text Recognition with Permuted Autoregressive Sequence Models*.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*.
- Cai, X., Wang, X., Zhu, Y., and Yao, Y. (2024). *MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting*.
- Gu, A., and Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*.
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*.
- Oreshkin, B. N. et al. (2020). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting*.
- Ahamed, M. A., and Cheng, Q. (2024). *TimeMachine: A Time Series is Worth 4 Mambas for Long-Term Forecasting*.
- Dao, T., and Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*.
- Shi, B., Bai, X., and Yao, C. (2017). *An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition*.
- Shi, B. et al. (2018). *ASTER: An Attentional Scene Text Recognizer with Flexible Rectification*.
- Liu, Y. et al. (2024). *VMamba: Visual State Space Model*.
- Zhu, L. et al. (2024). *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model*.
- Xu, K. et al. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*.
- Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?*

Use the paper landing pages/DOIs in the final bibliography, verify author lists and venue formatting before submission, and do not claim a paper introduced an exact implementation feature unless the paper supports that claim.
