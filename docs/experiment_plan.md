# Group 12 experiment plan

## 0. Decision summary and scope

**Task 1–2 dataset:** [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), UCI Machine Learning Repository. Use the original 370-customer, 15-minute load table, aggregate each customer to hourly mean load, and set `pred_column = MT_235`. The model may use the other stable-history customer columns as covariates, but the reported target is always `MT_235`.

**Task 3 dataset:** [SVHN, Format 1: Full Numbers](https://ufldl.stanford.edu/housenumbers/), not the cropped-digit format. Each original image has a `digitStruct.mat` annotation containing the ordered digit bounding boxes and labels. The target is the variable-length digit string in left-to-right order.

This pairing is deliberately distinct from the registered datasets listed in the brief: it is neither ETTh1, Hanoi/Open-Meteo weather, UCI Bike Sharing, CCPD, UFPR-AMR, nor multi-digit handwritten MNIST. The time-series task is not treated as a random IID regression problem: the central experiment is whether recurrent memory, temporal attention, selective state-space memory, and cross-customer context improve forecasts over seasonal and linear lag baselines. The image task is not treated as digit classification: it uses the full scene image and predicts the entire ordered number.

The report should make two claims only if the results support them:

1. Attention improves access to relevant daily/weekly history or spatial digit features, rather than being assumed to help.
2. The required RNN families remain competitive with strong modern comparators under a leakage-safe, repeated-seed protocol.

The main models are intentionally implementable by one student team on an RTX 3060, with the A100 reserved for repeated final runs and SOTA comparators.

## 1. Dataset choice

### 1.1 Task 1–2: primary dataset

**Dataset:** ElectricityLoadDiagrams20112014.

**Download and provenance.** The UCI landing page was checked on 2026-11-17 and is live; its Download control provides the archive. The data are 15-minute electricity consumption measurements for 370 customers from 2011–2014. The original SVHN and UCI pages are the authoritative links to put in the registration sheet, rather than a third-party mirror. The UCI page describes the file format and variables; preserve the original archive and record its SHA-256 after download.

**Audit-informed target and preprocessing for the experiment.**

The coder's audit found that `MT_001` is exactly zero for all of 2011 and only becomes live in 2012. That would make a chronological split deceptively easy and would violate the intent of a meaningful forecasting comparison. The target is therefore permanently changed to `MT_235`: it belongs to the 154 customers with full-history coverage and fewer than 1% zero hours, and it was among the audited hardest targets for persistence (`R²=0.27` for the audited last-15%-of-live-period test). This target choice is frozen before final model training; no test metric from the final experiment may be used to change it.

- Parse the semicolon-separated, decimal-comma table and its timestamp column. Convert the 15-minute measurements to hourly means, yielding exactly 35,064 hourly timestamps from 2011-01-01 through 2014-12-31 after the audit's row convention.
- Use `MT_235` as the fixed target. The main covariate set is the 154-client stable-history set identified by the audit: clients nonzero from the beginning of the raw period and with fewer than 1% zero hourly means. This removes structural pre-activation zeros from late-start clients. Freeze the client list in the run manifest and assert that it includes `MT_235`; if the implementation recomputes eligibility, it may use the training interval only and must report the resulting list.
- Add known-at-forecast-time calendar features: `sin/cos(hour/24)` and `sin/cos(day_of_week/7)`. Do not use calendar features that encode future observations.
- The main model input is the 370 hourly customer loads plus the four calendar features. If memory profiling shows that all 370 channels are unnecessarily slow, use the 32 highest-absolute-correlation customer channels selected using **training timestamps only**, and document the exact selected columns. This is a pre-specified engineering fallback, not a test-driven choice.
- Check timestamp regularity, duplicate timestamps, missing values, negative loads, constant columns, and zero-heavy columns before windowing. Short gaps of at most two hours may be linearly interpolated using adjacent observations; longer gaps must be removed from usable windows and counted. Never interpolate using validation or test values when constructing a training feature.
- Fit the feature standardization parameters on the training interval only. Inverse-transform predictions before all reported metrics.
- Main input window is `L = 168` hourly steps (one week). This is long enough for daily and weekly periodicity without making the recurrent model impractical. Task 1 predicts `y[t+2]` exactly; Task 2 predicts `[y[t+1], ..., y[t+5]]`.

**Why this is a good deep-learning test.** The audited target has a large scale (`mean≈792` kW), full history, and substantially weaker persistence than the easiest customers. The load has multiple time scales (intraday, weekday/weekend, weekly), customer-specific amplitude and regime changes, and cross-customer demand information. A target-only linear model with a few recent lags is an important baseline but is not allowed to stand in for a seasonal multivariate model. The one-week context gives attention and a selective SSM something meaningful to retain or suppress: a recent same-hour day, a weekday analogue, or a recent local transition. The design also permits an honest ablation between target-only, multivariate, calendar-aware, recurrent, and Mamba-based models.

**Two alternatives.**

| Alternative | Verified link | Why it was not selected |
|---|---|---|
| PeMSD7(M) traffic speed | [STGCN dataset directory / PeMSD7_Full.zip](https://github.com/VeritasYin/STGCN_IJCAI-18/tree/master/dataset) and the direct [ZIP](https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip) | Strong spatial-temporal benchmark, but it is only about two months of weekday 5-minute data. A single-sensor RNN study would underuse the spatial structure, while a graph model would expand the project beyond the stated RNN focus. |
| NYC TLC Yellow Taxi demand | [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Freely downloadable official Parquet files and highly nonstationary demand, but it requires a substantial aggregation pipeline, very large monthly files, and careful treatment of COVID-era regime changes. It is a good fallback if UCI is unavailable, not the best few-day project choice. |

**Why the primary wins:** UCI has a stable, documented, multi-year time axis; it is small enough to preprocess once; it naturally supports both a single future point and a five-step sequence; and its multiscale load structure makes the attention hypothesis testable without requiring a graph neural network.

### 1.2 Task 3: primary dataset

**Dataset:** SVHN Format 1, Full Numbers. Use `train.tar.gz` and `test.tar.gz`; each archive contains PNG images and `digitStruct.mat`. Do not use `train_32x32.mat`, `test_32x32.mat`, or `extra_32x32.mat`, because those are cropped single-digit recognition data and do not satisfy the intended image-to-sequence task as strongly as the full-number format.

**Download and provenance.** The [official Stanford page](https://ufldl.stanford.edu/housenumbers/) was checked on 2026-11-17. It explicitly lists the full-number train, test, and extra downloads and explains that `digitStruct.mat` contains each image name and the position, size, and label of each digit bounding box. The page also states that the data are for non-commercial use; this is compatible with a university course project, but the team must not redistribute the archive or claim commercial use.

**Preprocessing for the experiment.**

- Parse each MATLAB annotation, convert MATLAB label `10` to digit `0`, sort boxes by increasing `left` coordinate, and construct the target string. Retain original image-level identity; do not generate independent digit crops as samples.
- The audit found 33,402 train images and 13,068 test images. Train lengths are 5,137/18,130/8,691/1,434/9/1 for lengths 1–6; test lengths are 2,483/8,356/2,081/146/2 for lengths 1–5. Set `MAX_LEN=6`; retain the single six-digit training example and do not silently truncate it. The model must support the maximum observed training length even though the test maximum is five.
- Preserve aspect ratio, resize the full image to fit inside a `64 x 256` RGB canvas, and pad with the dataset mean. The audit's sampled width/height ranges justify a wide padded canvas; record the exact resize and padding policy. Save the parsed annotations and a deterministic split manifest.
- Use the official test archive only for the final test. Split the official train archive 90/10 into train/validation, stratifying by sequence length and using a fixed hash-based assignment so reruns reproduce the same split. Do not use the optional `extra` archive in the primary result; it has a different difficulty profile and would make comparison less clean.
- Training augmentation may use small horizontal translation, scale jitter, brightness/contrast jitter, and at most five degrees of rotation. Never flip horizontally, reorder crops, or apply augmentation that changes the digit string.

**Two alternatives.**

| Alternative | Verified link | Why it was not selected |
|---|---|---|
| IIIT 5K-Word | [official IIIT/CVIT project page](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset); the page lists the 106 MB download | Real word images and transcriptions make a clean image-to-sequence task, but only 5,000 cropped word images are available and the lexicon/image protocol can make the task partly a closed-vocabulary recognition benchmark. |
| Total-Text | [official GitHub dataset page](https://github.com/cs-chan/Total-Text-Dataset) | More varied natural text and curved layouts, but it is primarily an annotated scene-text detection dataset; extracting word images and transcriptions adds substantial preprocessing and annotation-format risk for a final project. |

**Why SVHN wins:** it is a real, freely downloadable, well-documented image-to-variable-length-digit problem; the bounding boxes make the sequence order unambiguous; it is large enough for a CNN+RNN model; and it supports both an attention decoder and a standard CTC/CRNN comparator.

## 2. Research questions and model comparisons

### 2.1 Task 1: two-step-ahead recurrent forecast

**Question:** Does a recurrent encoder with learned temporal attention and multivariate customer context improve `MT_235` prediction at `t+2` over persistence, seasonal, linear, and attention-free recurrent baselines, and does a selective state-space encoder add value beyond a matched recurrent encoder?

**Main model: Attentive BiGRU-2.**

- Input: 168 hourly observations, the frozen 154 stable-history load channels plus four calendar features.
- Encoder: two-layer GRU, hidden size 128 per direction, dropout 0.20 between recurrent layers. Use the sequence of encoder outputs, not only the final state.
- Attention: additive/Bahdanau temporal attention with a 128-unit scoring MLP over all 168 encoder states. The context vector is concatenated with the final forward/backward state.
- Head: LayerNorm, linear 256-to-128, GELU, dropout 0.20, linear 128-to-1.
- Loss: MSE on standardized `MT_235`; report the final result after inverse transformation. A Huber-loss sensitivity run is optional if extreme outliers dominate.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`, gradient clipping at 1.0, batch size 256, maximum 50 epochs, early stopping patience 8 on validation RMSE, reduce learning rate by 0.5 after three stagnant validation epochs.
- Seeds: exactly `0, 1, 2` for every run; use the same three seeds for main models, baselines, Mamba models, and all ablations. A smoke test may use fewer epochs, but it is not a final run.

The design is SOTA-informed but requirement-compliant. It uses the recurrent sequence encoder and attention pattern established for sequence-to-sequence learning, while the multivariate context and calendar variables reflect the variable-selection and interpretable temporal-weighting ideas of **“Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting” (Lim et al., 2021)**. Comparators include **“N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting” (Oreshkin et al., 2020)**, **“Are Transformers Effective for Time Series Forecasting?” (Zeng et al., 2023)**, and **“A Time Series is Worth 64 Words: Long-term Forecasting with Transformers” (Nie et al., 2023)**. The novel project contribution is a controlled, matched comparison of recurrent attention and selective state-space memory at the exact short horizon required by the course, with repeated seeds and formal paired tests.

**Additional Mamba method: Mamba-2–GRU hybrid with temporal attention.** This method first projects the 154-channel input to `d_model=128`, applies two residual Mamba-2 blocks in a forward and reverse scan, applies a one-layer GRU readout (`hidden=128`) to retain an explicit RNN component, and applies the same additive temporal attention and scalar head as the main model. The Mamba branch uses `d_state=16`, `d_conv=4`, `expand=2`, RMSNorm, and dropout 0.20. The forward/reverse outputs are projected back to 128 dimensions before the GRU. It therefore satisfies the course's RNN requirement rather than replacing the required RNN with an SSM.

This choice is grounded in **“Mamba: Linear-Time Sequence Modeling with Selective State Spaces” (Gu and Dao, 2023)** and the more efficient **“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality” (Dao and Gu, 2024)**. Its time-series adaptation is motivated by **“MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting” (Cai et al., 2024)** and **“TimeMachine: A Time Series is Worth 4 Mambas for Long-Term Forecasting” (Ahamed and Cheng, 2024)**. The project does not claim to reproduce MambaTS; it tests whether selective state updates help this shorter, multivariate electricity problem.

**Baselines and SOTA-informed comparators.** The locked execution set is the Phase 1/Phase 2 matrix in Section 4.4. The following literature comparators motivate the design but are not locked runs under the user's two-phase decision:

1. Persistence: `y[t+2] = y[t]`.
2. Seasonal naive: `y[t+2] = y[t+2-24]` and `y[t+2] = y[t+2-168]` where available.
3. Ridge regression on lags `{1, 2, 3, 24, 48, 168}` plus calendar features; choose regularization on validation only.
4. Target-only GRU-2 without attention.
5. Multivariate GRU-2 without attention.
6. Main Attentive BiGRU-2.
7. Mamba-2–GRU hybrid with temporal attention.
8. DLinear, N-BEATS, and PatchTST remain literature context for why the Mamba comparison is modern; they are not executed before the report handoff.

The primary Mamba comparison is the hybrid versus a parameter-matched Attentive BiGRU with the same GRU readout, attention module, input, training budget, and seeds. The primary course comparison is the Attentive BiGRU versus the attention-free multivariate GRU and best non-neural baseline. The SOTA comparators must not displace the required RNN experiments.

### 2.2 Task 2: five-step encoder–decoder forecast

**Question:** Does explicit encoder–decoder attention reduce error across the five future horizons and reduce error accumulation compared with direct linear, recurrent, and modern long-horizon baselines?

**Main model: Bahdanau Attentive LSTM Encoder–Decoder.**

- Input: the same 168-hour history and features as Task 1; output length exactly `H=5`.
- Encoder: two-layer LSTM, hidden size 128, dropout 0.20. Retain all encoder outputs and the final hidden/cell states.
- Decoder: one-layer LSTM, hidden size 128, initialized from the encoder final states. At each horizon, input the previous target value, the known future calendar embedding, and a learned horizon embedding for `1..5`.
- Attention: additive attention over all encoder outputs at every decoder step. Report the five attention maps.
- Output: one scalar per decoder step. Train with teacher forcing ratio 0.70, linearly reduced to 0.30 over the first 25 epochs; at validation/test use fully autoregressive decoding.
- Loss: mean MSE over the five standardized target values, with equal horizon weights for the primary result. Add a horizon-weighted sensitivity `(1, 1, 1, 1, 1.2)` only as an analysis, not as the main comparison.
- Optimizer: AdamW, learning rate `5e-4`, weight decay `1e-4`, gradient clipping 1.0, batch size 128, maximum 60 epochs, patience 10, and exactly seeds `0, 1, 2`.

This model follows **“Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation” (Cho et al., 2014)** and **“Neural Machine Translation by Jointly Learning to Align and Translate” (Bahdanau et al., 2015)** for recurrent encoder–decoder and learned alignment, then adapts them to multi-horizon load forecasting. It is informed by **“Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting” (Lim et al., 2021)** because it conditions each future horizon on known covariates and analyzes temporal importance. The strongest modern comparators are N-BEATS (2020), DLinear (2023), and PatchTST (2023). Do not call the LSTM itself SOTA; call it an interpretable, requirement-compliant recurrent model tested against SOTA-informed and Mamba-based baselines.

**Additional Mamba method: Mamba-2 encoder plus attentive LSTM decoder.** Replace only the two-layer LSTM encoder with two residual Mamba-2 blocks (`d_model=128`, `d_state=16`, `d_conv=4`, `expand=2`); retain a one-layer LSTM decoder, teacher forcing schedule, future-calendar inputs, horizon embeddings, Bahdanau attention, loss, and output head exactly as in the main model. Initialize the LSTM decoder state from a learned projection of the final Mamba state. This is still an encoder–decoder network and still contains the required RNN/LSTM decoder. The Mamba encoder is causal over the observed history; no future target values enter it.

The rationale is the same selective-memory lineage: Mamba (2023), Mamba-2/SSD (2024), MambaTS (2024), and TimeMachine (2024). MambaTS is particularly relevant because it studies variable-aware scans and selective dropout for multivariate long-horizon forecasting. Here the horizon is only five, so the experiment is a targeted test of whether the selective state update is useful even when a Transformer is not computationally necessary.

**Baselines and comparators.**

- Five-step persistence, daily seasonal, and weekly seasonal naive forecasts.
- Ridge direct multi-output regression on the same lag set.
- Recursive one-layer GRU without decoder attention.
- LSTM encoder–decoder without attention.
- Main Attentive LSTM encoder–decoder.
- Mamba-2 encoder + attentive LSTM decoder.
- N-BEATS, DLinear, and PatchTST are literature-only comparators in this execution plan; they are not run before the report handoff.

Report both pooled metrics and metrics separately at horizons 1–5. The key failure mode to analyze is a model that has a good average but deteriorates sharply at horizon 5.

### 2.3 Task 3: CNN encoder plus recurrent attention decoder

**Question:** Can a CNN+LSTM decoder with spatial visual attention recover ordered variable-length house numbers from full SVHN images, and does attention improve over global pooling and CTC alternatives?

**Main model: ResNet18/Bahdanau visual-attention decoder.**

- Encoder: ImageNet-initialized ResNet-18 up to the last convolutional feature map, with the classification head removed. If the dependency or licensing environment makes pretrained weights unavailable, use a five-block CNN with channels 64, 128, 256, 256, 256 and record that change.
- Convert the spatial feature map to a sequence of visual tokens in raster order. Keep the 2-D grid coordinates as learned row/column embeddings.
- Decoder: two-layer LSTM, hidden size 256, with additive visual attention over all spatial tokens at every character step. Vocabulary is digits 0–9 plus `<SOS>`, `<EOS>`, and `<PAD>`.
- Training: cross-entropy with teacher forcing ratio 1.0 for five epochs, then linearly decay to 0.50 by epoch 30. Label smoothing 0.05. AdamW at `3e-4`, weight decay `1e-4`, batch size 64 on the RTX 3060 or 128 on A100, maximum 40 epochs, patience 7, gradient clipping 1.0.
- Decode with greedy decoding for the primary reproducible result and beam size 3 as a secondary result. Stop at `<EOS>` or `MAX_LEN`.
- Use only label-preserving augmentation. Freeze the first ResNet stage for the first three epochs, then unfreeze all layers with a ten-times-smaller learning rate for the CNN than for the decoder.

This is SOTA-informed by **“Show, Attend and Tell: Neural Image Caption Generation with Visual Attention” (Xu et al., 2015)**, which established spatial attention for image-to-sequence generation; **“An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition” (Shi, Bai, and Yao, 2017)**, which established the CNN–recurrent sequence-recognition pattern; and **“ASTER: An Attentional Scene Text Recognizer with Flexible Rectification” (Shi et al., 2018)**, an attention-based scene-text sequence recognizer. A full ASTER or ABINet reimplementation is not required. The project contribution is a clean, auditable comparison between visual attention, CTC, and pooled recurrent decoding under the course’s four regression metrics, plus sequence accuracy/edit distance as necessary task metrics.

**Additional Mamba method: CNN + bidirectional Vision-Mamba token mixer + attentive LSTM decoder.** Keep the CNN and the two-layer LSTM visual-attention decoder, but insert two residual bidirectional Mamba blocks between the CNN feature map and the decoder. First average-pool the CNN feature map over height to form an ordered width-token sequence; add learned horizontal position embeddings; scan left-to-right and right-to-left; concatenate and project the two directions back to 256 channels. Use `d_state=16`, `d_conv=4`, `expand=2`, and the same dropout and optimizer budget as the main model. The LSTM decoder and Bahdanau visual attention remain mandatory, so this is still a CNN encoder plus RNN/LSTM decoder rather than a pure Mamba recognizer.

This is an OCR-specific adaptation of **“Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model” (Zhu et al., 2024)** and **“VMamba: Visual State Space Model” (Liu et al., 2024)**, which address the position sensitivity of visual tokens and bidirectional/2-D selective scanning. The original **“Mamba: Linear-Time Sequence Modeling with Selective State Spaces” (Gu and Dao, 2023)** supplies the selective state-space mechanism. The project novelty is a controlled test of whether a selective visual token mixer improves full-number recognition while retaining the required recurrent decoder; it is not a claim to have invented a new OCR architecture.

**Baselines and comparators.**

1. CNN global-average-pooling + one-layer LSTM decoder, no visual attention.
2. Main CNN spatial-attention + two-layer LSTM decoder.
3. CRNN: CNN feature map collapsed along height, bidirectional LSTM, linear character head with CTC loss; this is the standard sequence-recognition comparator from the 2017 CRNN paper.
4. CNN + bidirectional Vision-Mamba token mixer + attentive LSTM decoder.
5. Same main model with a randomly initialized CNN, to measure transfer-learning contribution.
6. A lightweight PARSeq-style Transformer decoder, citing **“Scene Text Recognition with Permuted Autoregressive Sequence Models” (Bautista and Atienza, 2021)**, as a secondary non-recurrent comparator only after all required models are complete.

Primary image metrics are computed on the decoded sequence, not on teacher-forced logits. Also report exact full-string accuracy, character accuracy, normalized Levenshtein distance, and length accuracy; these are necessary because four numeric regression metrics do not fully describe OCR quality.

## 3. Ablation matrix

Every ablation below uses the same split, preprocessing, optimizer search budget, early-stopping rule, and checkpoint policy as its parent model. Mamba claims are evaluated with same-seed paired predictions: seed `s` for a Mamba model uses the same data split, batch order, augmentation stream, and early-stopping budget as seed `s` for its matched control. Widths are tuned before the final run so trainable parameter counts differ by at most 5%; report the exact counts. A single seed is acceptable for smoke tests only, never for a final Mamba claim.

### Task 1 ablations

| ID | Variant | What it tests |
|---|---|---|
| T1-A0 | Main Attentive BiGRU-2 | Full model |
| T1-A1 | Remove temporal attention; use final bidirectional state | Whether attention matters |
| T1-A2 | Target-only input | Whether cross-customer context matters |
| T1-A3 | All customer features, no calendar features | Whether known periodic covariates matter |
| T1-A4 | `L=24` instead of `L=168` | Whether weekly context matters |
| T1-A5 | One-layer GRU, hidden 64 | Capacity control |
| T1-A6 | LSTM encoder instead of GRU | Recurrent-cell sensitivity |
| T1-M0 | Mamba-2–GRU hybrid with attention | Proposed Mamba method |
| T1-M1 | Parameter-matched BiGRU encoder + same GRU readout and attention | Matched non-Mamba control |
| T1-M2 | Mamba static-state control: replace input-dependent `Δ/B/C` selection with learned time-invariant state parameters, retaining equal-width projection/FFN capacity | Whether selective input-dependent state updates matter |
| T1-M3 | Mamba without depthwise local convolution (`d_conv=1`) with compensating projection width | Whether the local convolution sub-part matters |
| T1-M4 | Forward-only Mamba scan instead of forward+reverse scan | Whether bidirectional state aggregation matters |

The decisive Task 1 Mamba evidence is T1-M0 versus T1-M1 and T1-M2. T1-M3/T1-M4 explain which Mamba sub-parts drive any gain; they are not allowed to be omitted if the project claims that selective state memory, rather than just extra parameters, caused the improvement.

### Task 2 ablations

| ID | Variant | What it tests |
|---|---|---|
| T2-A0 | Main attentive LSTM encoder–decoder | Full model |
| T2-A1 | Remove decoder attention; use encoder final state only | Whether alignment helps |
| T2-A2 | Teacher forcing fixed at 1.0 | Scheduled-sampling contribution and exposure bias |
| T2-A3 | Recursive one-step GRU baseline | Error accumulation from autoregression |
| T2-A4 | `L=24` instead of `L=168` | Weekly context contribution |
| T2-A5 | Target-only input | Cross-customer contribution |
| T2-A6 | Remove future calendar and horizon embeddings | Known-future information contribution |
| T2-M0 | Mamba-2 encoder + attentive LSTM decoder | Proposed Mamba method |
| T2-M1 | Parameter-matched LSTM encoder + identical attentive LSTM decoder | Matched non-Mamba control |
| T2-M2 | Mamba static-state control with input-independent `Δ/B/C` | Whether selective updates matter |
| T2-M3 | Mamba encoder without depthwise local convolution with matched projection capacity | Local convolution contribution |
| T2-M4 | Mamba encoder with attention removed from the decoder | Interaction between Mamba memory and decoder alignment |

The decisive Task 2 Mamba evidence is T2-M0 versus T2-M1 and T2-M2, reported both pooled and at each horizon. T2-M3 and T2-M4 are mechanism checks, not optional tuning variants.

### Task 3 ablations

| ID | Variant | What it tests |
|---|---|---|
| T3-A0 | CNN + spatial-attention LSTM | Full model |
| T3-A1 | Global average pooling + LSTM | Visual attention contribution |
| T3-A2 | CRNN + CTC | Autoregressive decoder versus alignment-based decoding |
| T3-A3 | Random CNN initialization | ImageNet transfer contribution |
| T3-A4 | No augmentation | Robustness contribution |
| T3-A5 | One-layer decoder, hidden 128 | Capacity control |
| T3-A6 | Greedy versus beam-3 decoding | Decoding contribution |
| T3-M0 | CNN + bidirectional Vision-Mamba token mixer + attentive LSTM decoder | Proposed Mamba method |
| T3-M1 | CNN + parameter-matched bidirectional GRU token mixer + identical attentive LSTM decoder | Matched non-Mamba visual sequence control |
| T3-M2 | Vision-Mamba static-state control with input-independent selection parameters | Whether selective visual state updates matter |
| T3-M3 | Forward-only visual Mamba scan | Whether bidirectional visual scanning matters |
| T3-M4 | Remove learned horizontal position embeddings, keeping the same Mamba blocks | Whether positional marking is essential for ordered digits |

The decisive Task 3 Mamba evidence is T3-M0 versus T3-M1 and T3-M2, using exact-string accuracy and the required numeric metrics. T3-M3/T3-M4 isolate the two visual adaptations motivated by Vision Mamba: bidirectional context and explicit position marking.

### 3.1 What counts as evidence that Mamba works

For each task, pre-register the Mamba claim as: the Mamba method must have lower mean test RMSE than its matched control, and the 95% paired bootstrap interval for the primary loss difference must exclude zero in the favorable direction. Every contrast uses exactly seeds `0, 1, 2`; if the interval includes zero, write “no reliable Mamba improvement” rather than selecting a favorable seed. Use the same criterion for the Mamba-vs-static-state comparison. This is stronger than comparing a single best run and directly tests the selective component.

The bootstrap resamples test origins in blocks for Tasks 1–2 and test images for Task 3. Also report the three paired seed-level differences and their mean/std, but do not use a t-test across seeds: `n=3` is too small for that claim. Mamba ablations must use the same model budget and parameter-count tolerance, not a larger hidden size or more epochs.

## 4. Evaluation protocol and leakage controls

### 4.1 Splits

**Time series:** chronological 70% train, 15% validation, 15% test by hourly timestamp. Construct a window only when the forecast target lies wholly in its split. It is acceptable for a validation/test input window to use historical context immediately before the split boundary, because that is available at prediction time; it is not acceptable to use any future value. Fit scalers, feature selection, missing-value rules, and all hyperparameters using train/validation only. Test is run once after model selection.

**SVHN:** official test archive is untouched. Split official train 90/10 into train/validation with a fixed hash manifest, stratified by sequence length. Any duplicate-image audit must happen before splitting; duplicates or near-duplicates crossing train and validation/test are removed from training and reported. Do not use the extra split for the headline result.

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

For SVHN, use a paired bootstrap over test images with 10,000 replicates for the difference in exact-string accuracy, character accuracy, MAE, and RMSE. Average the same-seed per-image loss differences before the bootstrap for M0-M1 and M0-M2. For exact-string correctness, add McNemar's test between the Mamba method and matched control and between the course main and strongest baseline. Apply Holm correction to the two primary Mamba hypotheses. Use two-sided tests and state that statistical significance is secondary to effect size and confidence intervals.

With exactly three seeds, the primary image test is the paired bootstrap over per-image differences after averaging the three same-seed correctness/loss values. Do not pretend that the 3 x 13,068 seed-image pairs are independent. McNemar's test may be reported separately for each seed as a descriptive secondary check, but the claim-supporting interval is the clustered per-image bootstrap; no seed-level t-test is allowed.

Do not claim that a p-value proves one deep architecture is universally superior. The valid claim is conditional on this dataset, horizon, preprocessing, and model budget.

### 4.4 Locked final run matrix and A100 budget

This is the final matrix, not a menu. A short smoke run is allowed before it, but it cannot replace the listed runs. Every listed run uses exactly seeds `0, 1, 2`. Phase 1 must finish and export its tables before any Phase 2 run starts. The modern N-BEATS/DLinear/PatchTST papers remain literature context, but those extra models are not in this locked execution matrix because the user explicitly ordered Phase 1 to contain only report-required baselines/course models and Phase 2 to contain only Mamba proposals/ablations.

| Phase | Task | Model family | Seeds | Runs | Estimated active A100 hours |
|---|---:|---|---:|---:|---:|
| 1 | 1 | Persistence, daily seasonal naive, weekly seasonal naive, ridge | 0,1,2 | 12 model/seed evaluations | 0.0 |
| 1 | 1 | Required Attentive BiGRU-2 and no-attention multivariate GRU | 3 each | 6 | 3.0 |
| 1 | 2 | Persistence, daily seasonal naive, weekly seasonal naive, direct multi-output ridge | 0,1,2 | 12 model/seed evaluations | 0.0 |
| 1 | 2 | Required attentive LSTM encoder–decoder and no-attention LSTM encoder–decoder | 3 each | 6 | 4.2 |
| 1 | 3 | Required CNN spatial-attention LSTM, global-pooling LSTM counterpart, and CRNN-CTC baseline | 3 each | 9 | 12.6 |
| 1 | all | Smoke runs, checkpoint/resume validation, and Phase 1 export verification | — | — | 2.0 |
| 2 | 1 | Mamba-2–GRU hybrid T1-M0 and matched attentive BiGRU T1-M1 | 3 each | 6 | 3.0 |
| 2 | 1 | Mamba static-state, no-local-conv, forward-only T1-M2–M4 | 3 each | 9 | 3.6 |
| 2 | 2 | Mamba-2 encoder + attentive LSTM decoder T2-M0 and matched LSTM encoder T2-M1 | 3 each | 6 | 4.2 |
| 2 | 2 | Static-state, no-local-conv, no-decoder-attention T2-M2–M4 | 3 each | 9 | 4.5 |
| 2 | 3 | CNN + Vision-Mamba token mixer T3-M0 and matched BiGRU token mixer T3-M1 | 3 each | 6 | 9.6 |
| 2 | 3 | Static-state, forward-only, no-position T3-M2–M4 | 3 each | 9 | 11.7 |
| 2 | all | Smoke runs, checkpoint/resume validation, and Phase 2 export verification | — | — | 1.7 |
| **Total** |  |  |  | **90 model/seed evaluations** | **60.1 active hours** |

CPU-only persistence, daily/weekly seasonal naive, ridge, metric calculations, DM tests, and bootstrap calculations do not consume meaningful A100 time. Reserve **75 A100 wall-clock hours**: approximately 60 active training/evaluation hours plus 15 hours for preemption, resume, compilation, and queue delay. Do not add PARSeq, N-BEATS, DLinear, PatchTST, extra hyperparameter searches, or non-Mamba ablations before the report is complete. The must-have scientific result is the Phase 1 report table followed by the Phase 2 matched Mamba/control/static-state matrix with seeds `0, 1, 2`.

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

For image `i`, let the ground-truth digit sequence be `y_i = (y_{i,1},...,y_{i,L_i})`, with `y_{i,j} in {0,...,9}`. Decode the model autoregressively, remove `<SOS>`, stop at `<EOS>`, and let `hat{y}_{i,j}` be the decoded digit at position `j`.

The primary numeric convention is a **fixed-position, length-aware padded representation**:

- Evaluate positions `j = 1..MAX_LEN`.
- For true positions `j <= L_i`, use the decoded digit if present; if the decoder emits `<EOS>` too early, assign a missing-digit prediction of `0` for that position so early stopping is penalized.
- For `j > L_i`, do not include the position in R2/NSE/MAE/RMSE; PAD is not a numeric digit and must never enter these metrics.
- If the decoder emits extra digits after the true end, count them in the separate length/edit metrics, not as numeric target digits.

Let `M = sum_i L_i` be the number of valid target digit positions and let `z_{i,j}` be the padded/aligned prediction for those positions:

```text
MAE  = (1/M) * sum_i sum_{j=1..L_i} |z_{i,j} - y_{i,j}|
RMSE = sqrt((1/M) * sum_i sum_{j=1..L_i} (z_{i,j} - y_{i,j})^2)
R2   = 1 - sum_i sum_{j=1..L_i} (z_{i,j} - y_{i,j})^2
           / sum_i sum_{j=1..L_i} (y_{i,j} - mean_valid(y))^2
NSE  = 1 - sum_i sum_{j=1..L_i} (z_{i,j} - y_{i,j})^2
           / sum_i sum_{j=1..L_i} (y_{i,j} - mean_valid(y))^2
```

This is the primary convention because it is explicit about variable length, penalizes early termination, and does not assign an artificial numeric value to PAD. Again, R2 and NSE are identical under the shared denominator. Also report these secondary sequence conventions:

1. **Position macro-average:** calculate metrics for each position over images that have a true digit at that position, then average positions. This prevents the common first digit from dominating, but has higher variance for rare positions.
2. **Sample macro-average:** calculate metrics per image over its valid digits, then average images. This gives every number equal weight.
3. **Fixed-width padded:** encode EOS/PAD as categorical states and report classification accuracy, not R2/NSE/MAE/RMSE. This is useful for diagnosing length errors but is not the primary required metric.

The report must not present R2/NSE as sufficient OCR metrics. Add exact full-sequence accuracy, exact length accuracy, per-character accuracy over aligned valid positions, and normalized Levenshtein distance. These identify cases where numeric digit error is small but the whole number is still wrong.

## 6. Analysis experiments and report figures

The following figures are enough for a strong report; avoid producing decorative plots without a question.

1. **Dataset overview:** one week of stable-client aggregate load and `MT_235`, plus a seasonal profile by hour and weekday. Include the audit's exact row count, 154-client covariate count, structural-zero counts, missingness, and timestamp-regularity checks.
2. **Autocorrelation and baseline difficulty:** ACF at lags 1, 24, and 168; scatter of persistence/seasonal baseline error by hour of day. This motivates the recurrent context and shows whether the problem is nontrivial.
3. **Task 1 forecast panel:** representative test windows showing true `MT_235`, persistence, best linear baseline, attention-free GRU, attentive BiGRU, and Mamba-2–GRU; include a zoomed two-step forecast.
4. **Task 1 Mamba ablation plot:** mean ± standard deviation across exactly three seeds for M0/M1/M2/M3/M4, with paired-bootstrap intervals for Mamba versus matched BiGRU and static-state controls.
5. **Task 1 attention/state heatmaps:** temporal attention weights and Mamba selection/state diagnostics over the 168-hour input for several test cases, with markers at 24-hour and 168-hour lags. Include a failure case, not only a success case.
6. **Task 2 horizon plot:** MAE and RMSE at horizons 1–5 for all key models, including Mamba encoder–decoder and matched LSTM. This is the main evidence for error accumulation and whether selective memory helps at later steps.
7. **Task 2 attention/state examples:** encoder-time versus decoder-horizon attention heatmaps, plus Mamba-vs-static-state diagnostics; discuss whether different horizons use different historical positions.
8. **Task 3 qualitative grid:** correct and incorrect SVHN predictions from course-main, Mamba, and matched visual-GRU models, showing input image, ground-truth string, prediction, and attention overlay for each decoded character.
9. **Task 3 Mamba ablation/error breakdown:** exact-sequence accuracy by number length, character accuracy by position, M0/M1/M2 confidence intervals, and separate early-EOS, wrong-digit, extra-digit, and position-embedding failures.
10. **Compute/accuracy trade-off:** parameter count, training time, and test RMSE/sequence accuracy for the main models and comparators. This prevents a small improvement from being presented without cost.

Required tables: dataset audit statistics and splits; complete model hyperparameters and parameter counts; primary metrics with mean ± standard deviation over seeds; per-horizon Task 2 metrics; Task 3 numeric and OCR metrics; Mamba ablations with adjusted p-values and confidence intervals; and the final run/checkpoint manifest.

## 7. Two-phase execution order and coder TODO

The coder must not start Phase 2 until every Phase 1 run has finished, its checkpoints and predictions are saved, and its report tables have been exported. Every final run in both phases uses exactly seeds `0, 1, 2`.

### Phase 0: locked data and export setup

1. Download the UCI and SVHN files, record checksums, parse the audited decimal-comma UCI file, and verify the exact row/image counts.
2. Freeze `MT_235`, the 154-client stable-history covariate list, the chronological time-series splits, the SVHN train/validation hash split, and `MAX_LEN=6`.
3. Implement leakage-safe windowing, train-only scaling, inverse transforms, the four required metric functions, masked sequence metrics, and hand-worked unit tests.
4. Implement checkpoint/resume with model, optimizer, scheduler, scaler, RNG, epoch, and data-order state. Validate resumption on the shared A100 before any final run.

Phase 0 is CPU/I/O work plus at most 1 A100 hour for a resume smoke test. It exports `manifests/dataset_manifest.json`, `manifests/split_manifest.json`, `manifests/environment.json`, and `manifests/checksums.txt`.

### Phase 1: report-required baselines and course models only

Phase 1 is intentionally limited to the models needed for the template's experimental-results tables:

- **Task 1:** persistence, daily seasonal naive, weekly seasonal naive, ridge; required Attentive BiGRU-2; multivariate GRU-2 without attention.
- **Task 2:** persistence, daily seasonal naive, weekly seasonal naive, direct multi-output ridge; required attentive LSTM encoder–decoder; LSTM encoder–decoder without decoder attention.
- **Task 3:** required CNN spatial-attention LSTM; global-pooling CNN-LSTM no-attention counterpart; CRNN-CTC baseline.

Run every Phase 1 model or baseline under exactly seeds `0, 1, 2`, export all Phase 1 tables, and stop. For deterministic CPU baselines, execute the same three-seed evaluation jobs and write one row per seed even when predictions are identical. Do not run Mamba, Mamba ablations, N-BEATS, DLinear, PatchTST, PARSeq, random-CNN controls, or exploratory non-Mamba ablations in Phase 1. The expected Phase 1 A100 allowance is approximately 22 active hours, including smoke/resume validation; CPU baselines and statistics are outside that estimate.

Required Phase 1 deliverables:

- `results/phase1/task1_metrics_by_seed.csv` and `results/phase1/task1_metrics_summary.csv` with R2, NSE, MAE, RMSE for every baseline/model and seed.
- `results/phase1/task2_metrics_by_seed.csv`, `task2_metrics_summary.csv`, and `task2_metrics_by_horizon.csv` with pooled and horizon-wise metrics.
- `results/phase1/task3_metrics_by_seed.csv` and `task3_metrics_summary.csv` with R2, NSE, MAE, RMSE, exact-string accuracy, character accuracy, length accuracy, and normalized edit distance.
- `results/phase1/predictions/` containing one machine-readable prediction file per model/seed and the per-origin/per-sample error files required for paired comparisons.
- `figures/phase1/` containing the dataset overview, baseline forecast panels, Task 2 horizon plot, and Task 3 qualitative/error plots.

### Phase 2: Mamba proposals and Mamba ablations only

Start Phase 2 only after the Phase 1 export checklist passes. Run exactly the M0/M1/M2/M3/M4 models in Section 4.4 with seeds `0, 1, 2`:

- Task 1: Mamba-2–GRU hybrid, matched attentive BiGRU, static-state Mamba, no-local-convolution Mamba, and forward-only Mamba.
- Task 2: Mamba-2 encoder plus attentive LSTM decoder, matched LSTM encoder, static-state Mamba, no-local-convolution Mamba, and no-decoder-attention Mamba.
- Task 3: CNN plus Vision-Mamba token mixer plus attentive LSTM decoder, matched visual-BiGRU mixer, static-state visual Mamba, forward-only visual Mamba, and no-position visual Mamba.

Export the same Phase 1 metrics plus `results/phase2/mamba_ablation_metrics_by_seed.csv`, `mamba_ablation_summary.csv`, `paired_error_differences.csv`, `dm_tests.csv`, `bootstrap_confidence_intervals.csv`, and `holm_adjusted_tests.csv`. Export Mamba selection/state diagnostics and attention maps for the figures in `figures/phase2/`.

The locked Phase 2 allowance is approximately 38 active A100 hours. The total locked experiment budget is therefore approximately 60 active A100 hours. Reserve 75 A100 wall-clock hours for preemption, resume, compilation, and queue delay. Use `mamba-ssm` with the existing PyTorch 2.8/CUDA 12.9 environment; use `mambapy` only as a recorded fallback if compilation fails. Use the project virtual environment/`uv`, persistent storage, plain `nohup` for background jobs, and the pod keepalive procedure. Do not use `pm2`.

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
| `dataset_manifest.json` | Dataset name, authoritative download URL, raw filename, checksum, raw observations, hourly observations, all 370 original attributes, 154 stable-history attributes, frequency, time period, `pred_column=MT_235`, target mean/std/min/max/zero count, UCI parsing convention, SVHN image count, image format, resized canvas, sequence vocabulary, `MAX_LEN=6`, and train/test length histograms. |
| `split_manifest.json` | Exact train/validation/test timestamp boundaries and window counts for Task 1 and Task 2; exact SVHN train/validation/test image counts and sequence-length counts. Include the formulas `N-L-2+1` and `N-L-5+1` and the realized counts after boundary filtering. |
| `phase1/task1_model_specs.json` | Title, objective, input shape/window, output shape, architecture diagram data, parameter count, optimizer, learning rate, batch size, epochs, patience, loss, seed list, and checkpoint path for every Task 1 model. |
| `phase1/task2_model_specs.json` | The same fields for the encoder–decoder models, including teacher-forcing schedule, horizon embeddings, and attention configuration. |
| `phase1/task3_model_specs.json` | The same fields for CNN-LSTM and CRNN-CTC models, including image canvas, CNN feature-map size, decoder vocabulary, augmentation, teacher forcing, decoding mode, and parameter count. |
| `*_metrics_by_seed.csv` | One row per model/seed with R2, NSE, MAE, RMSE and, for Task 3, exact-string accuracy, character accuracy, length accuracy, normalized edit distance. |
| `task2_metrics_by_horizon.csv` | One row per model/seed/horizon 1–5 with R2, NSE, MAE, RMSE, plus pooled and macro summaries. |
| `mamba_ablation_summary.csv` | M0/M1/M2/M3/M4 means, standard deviations, medians, per-seed values, parameter counts, training hours, and the pre-registered claim status. |
| `paired_error_differences.csv` | Same-seed per-origin Task 1/2 losses and same-seed per-image Task 3 losses for M0-M1 and M0-M2; include model names, seed, origin/image ID, horizon, true value, prediction, and loss. |
| `dm_tests.csv` | DM statistic, HAC lag, loss type, raw p-value, Holm-adjusted p-value, contrast, task, and seed-averaging rule. |
| `bootstrap_confidence_intervals.csv` | 10,000-replicate block/image bootstrap intervals for RMSE, MAE, NSE, exact-string accuracy, and the Mamba contrasts. |
| `references.csv` | Citation key, exact title, authors, year, venue/DOI or URL, and which report section uses it. |
| `figure_index.json` | Figure filename, caption, task, source CSV/JSON, and the claim the figure supports. |

The code must also export these final report figures as PNG and editable SVG where possible:

1. `figures/dataset_overview.png`: UCI target/covariate load overview, hourly frequency, period, and ACF at lags 1/24/168.
2. `figures/task1_baseline_forecasts.png`: `MT_235` test forecasts for persistence, best seasonal/ridge, no-attention GRU, and Attentive BiGRU.
3. `figures/task1_mamba_ablation.png`: M0/M1/M2/M3/M4 RMSE/MAE with 95% paired intervals.
4. `figures/task2_horizon_metrics.png`: Task 2 RMSE/MAE/R2/NSE at horizons 1–5 for Phase 1 and Mamba models.
5. `figures/task2_attention_state_maps.png`: decoder attention and Mamba state/selection diagnostics.
6. `figures/task3_examples_attention.png`: SVHN input, ground truth, prediction, and per-character attention for correct and failed examples.
7. `figures/task3_length_errors.png`: exact-string accuracy by sequence length, character/length errors, and Mamba ablation intervals.
8. `figures/compute_accuracy_tradeoff.png`: parameter count, active training time, and primary test metric.

### Phase 4: report review and revision timeline

The timeline is: Phase 0 setup; Phase 1 required models and exported tables; Phase 2 Mamba models/ablations and paired statistics; Phase 3 `.docx` report construction in the exact template structure; then a review pass using the available academic-writing/paper-review skills, followed by revision of the report based on that review. The reviewed and revised `.docx`, not the first draft, is the handoff artifact.

**Operational stopping rule:** if the schedule becomes tight, finish Phase 1 exports and the Phase 2 matched Mamba/control/static-state matrix before adding any literature-only comparator or extra figure. The three-seed paired error analysis is the evidence standard.

## 8. References to cite in the eventual report

- Bahdanau, D., Cho, K., and Bengio, Y. (2015). *Neural machine translation by jointly learning to align and translate*.
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
