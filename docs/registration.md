Project Title - "Selective-State Memory for Electricity Demand and Street-View Number Reading"

Project Description - 1. For Task 1, we use the hourly-aggregated ElectricityLoadDiagrams20112014 dataset and predict the `MT_235` electricity-load value two hours ahead from a 168-hour input window using a multivariate bidirectional GRU with temporal attention. We additionally evaluate a Mamba-2–GRU hybrid with the same recurrent readout and attention, alongside persistence, seasonal, linear, attention-free recurrent, and matched selective-state ablations.

2. For Task 2, using the same ElectricityLoadDiagrams20112014 dataset and `MT_235` target, we use a 168-hour LSTM encoder and autoregressive LSTM decoder with Bahdanau attention to forecast the next five consecutive hourly load values. We additionally evaluate a Mamba-2 encoder with the same attentive LSTM decoder, comparing both methods with seasonal baselines, direct linear regression, a recurrent model without attention, and matched static-state ablations.

3. For Task 3, we use the full-number format of the Street View House Numbers (SVHN) dataset, in which each original street-view image is paired with an ordered variable-length digit sequence. We build a CNN encoder with a visual-attention LSTM decoder and additionally test a bidirectional Vision-Mamba visual token mixer before the LSTM decoder, comparing it with a pooled CNN-LSTM decoder, a CRNN-CTC baseline, a matched visual-GRU control, and selective-state ablations.

Link to Download Data - Task 1 & 2: ElectricityLoadDiagrams20112014 (https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) and Task 3: Street View House Numbers (SVHN), Format 1 Full Numbers (https://ufldl.stanford.edu/housenumbers/)
