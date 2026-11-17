# SVHN full-number recognition. Common protocol vs our setup

## Bottom line

- Almost every public SVHN multi-digit pipeline crops each image to the union of the digitStruct boxes. They do this at train time and at test time, and the test crop uses the test set's own ground-truth boxes. We feed the whole image, so we are solving a harder task than the one behind the 96% numbers.
- Our val result of 74.7% matches the published uncropped baseline. He et al. 2019 get 74.89% with soft attention on whole SVHN images, and they also trained on the extra set. So our val number is about what the uncropped task gives. Our test number of 56.0% is the real anomaly.
- To follow common practice, we should switch to a GT union-box crop with a 30% expansion, resize to 64x64, and train on a random 54x54 crop. The CNN + LSTM attention decoder can stay as it is. Adding the extra set is the second lever.
- The course requirements PDF has no rule against using test digitStruct boxes. I searched its text and it never mentions SVHN, boxes or cropping. The report should still say plainly that GT test crops assume perfect localization (an oracle).
- I could not read any Kaggle notebook directly. The pages render with JavaScript, and the Kaggle API was blocked. I read two Kaggle notebooks through their GitHub copies (amourab, shudhanshurp). Both use GT crops.

## 1. The common protocol

Counts below cover the 15 public code repos or notebooks whose preprocessing could be read or was described. Papers are listed separately.

| Choice | Most common setting | How common | Sources |
|---|---|---|---|
| Test-time crop | GT union box from the test digitStruct | 14 of 15 repos. The one exception crops from a predicted box and gets 51% | potterhsu (PyTorch, TF), q19316, lixuan0023/SVHN_RNN, caoquanjie, Genius-Society, k-chuang, hangyao, Packt, carysva, thomalm, amourab, shudhanshurp, pitsios. The exception is pavitrakumar78 |
| Train-time crop | Same GT union box | All repos and all papers in the cropped setting | Goodfellow 2014 Sec 5.1, DRAM, STN, EDRAM, Yousef, He et al. |
| Box shape | Rectangular union box. The potterhsu family squares it first (side = max(w,h), centered) | 5 square, 7 rectangular | Square in potterhsu x2, SVHN_RNN, caoquanjie, Genius. Rectangular in q19316, carysva, k-chuang, hangyao, Packt, thomalm, amourab |
| Margin | 30% total (0.15 x side on each side, 1.3x) | 8 repos plus Goodfellow. 35% in 1 (k-chuang). 20% in 2 (hangyao, Packt). About 10% x and 16% y in 1 (amourab) | Goodfellow says "expand this bounding box by 30% in both the x and the y direction". caoquanjie has an extra *1.5 factor that looks like a bug |
| Resize | 64x64 with aspect ratio NOT kept | 10 repos at 64x64. 4 at 32x32. 1 keeps aspect (amourab, 64x160 canvas) | Goodfellow, potterhsu, q19316, SVHN_RNN, carysva (then 96x96), shudhanshurp, pitsios. 32x32 in k-chuang, hangyao, Packt, thomalm |
| Crop augmentation | Random 54x54 from 64x64 at train, center 54x54 at test | potterhsu x2, caoquanjie, Genius, q19316 (RandomResizedCrop scale 0.7-1.0) and all the main papers. SVHN_RNN and Keras_SVHN skip it | Goodfellow says removing it costs "about half a percentage point" |
| Color | RGB for the strong repos. Grayscale for the 32x32 repos and the attention papers | RGB in 8 repos, grayscale in 4 repos plus DRAM, EDRAM, Yousef | DRAM says color does not change final performance |
| Normalization | No standard. Most map to about [-1,1] (Normalize 0.5/0.5, or (x-128)/128) | Mixed. Others use /255, z-score, per-image, or none (q19316) | For our pretrained ResNet, keep the ImageNet mean and std |
| Extra set | Used. train + extra pooled, random 10% held out as val | 9 repos use it, 3 do not (Packt, carysva, Genius), the rest do not say. All of the roughly 96% papers use it | DRAM states 200,000 train and 5,000 val from train+extra |
| Max length | Labels longer than 5 dropped (only 1 train image has 6 digits) | potterhsu family, hangyao, thomalm. amourab keeps 4 or fewer | This changes the metric slightly. Report how 6-digit labels are handled |
| Other augmentation | Little or none beyond the random crop | potterhsu uses only RandomCrop. q19316 adds ColorJitter 0.5. SVHN_RNN uses brightness, saturation and hue | |

The reference recipe is Goodfellow as implemented in potterhsu. Take the union box, square it, and expand it 1.3x. Resize to 64x64 RGB. Use RandomCrop 54 for training and CenterCrop 54 for evaluation, with Normalize 0.5/0.5. Train on train+extra with a random 10% for val. Training uses SGD with momentum 0.9 and weight decay 5e-4, and StepLR with gamma 0.9. The README run used batch 512, lr 0.16 and 54,000 steps, and reached 95.65%.

The repo closest to our model is q19316/Pytorch-Attention-RNN-for-SVHN. It pairs a VGG-BN encoder with a 2-layer GRU, additive attention and hidden size 256. It uses AdamW 3e-4, batch 64, 16 passes, lr x0.1 at epoch 10, grad clip 5, full teacher forcing, RandomResizedCrop(54, 0.7-1.0) and ColorJitter 0.5, trained on train+extra. It reaches about 96.2% exact-match test accuracy. It has no separate val split.

## 2. Typical sequence accuracy under that protocol

All numbers are exact-match accuracy on the 13,068-image test set, unless noted otherwise.

### GT crop, with extra

| Model | Type | Seq. acc. | Caveat |
|---|---|---|---|
| He et al. 2019, conditional att2 | CNN + LSTM + attention | 97.15% | Paper |
| He et al. 2019, standard soft attention | CNN + LSTM + attention | 96.47% | Paper |
| Yousef et al. 2018 | Gated FCN | 96.9% (3.1% error) | Paper, grayscale 32x32 |
| STN, ST-CNN Multi | CNN | 96.4% (3.6% error) | Paper |
| EDRAM Double / Single | Recurrent attention | 96.4% / 95.64% | Numbers taken from Yousef Table 2 |
| lixuan0023/SVHN_RNN | CNN + LSTM, no attention | 96.34% | 64-unit LSTM, about 6 days |
| q19316 | CNN + attention GRU | about 96.2% | README |
| DRAM, forward-backward MC | Recurrent attention | 96.1% (3.9% error) | Single DRAM 94.9% |
| caoquanjie | CNN multi-head | 96.04% | Best of many logged test evaluations, so selected on test |
| Goodfellow et al. 2014 | CNN multi-head | 96.03% | Char accuracy 97.84% |
| Paddle-SVHN | CNN multi-head | 95.71% in the table, 95.3551% in the eval log | The two do not agree |
| potterhsu PyTorch | CNN multi-head | 95.65% | Ignores the length head, compares 5 padded slots |
| Keras_SVHN | CNN + LSTM | 95.5% | Checkpoint chosen on test |
| k-chuang | CNN multi-head | 94.399% | 32x32 grayscale |
| potterhsu TF | CNN multi-head | 93.45% | |
| beeps82, pretrained VGG-16 | CNN | 91.24% | About 150k samples, setup not described |
| amourab (Kaggle) | EfficientNetV2B0 multi-head | 75.40% | Only 80k extra, aspect-kept 64x160, 4 digits max. Only 41.10% on 4-digit numbers |

### GT crop, train only (33k)

| Model | Seq. acc. | Caveat |
|---|---|---|
| Guo et al. 2016, CRNN + CTC | 91% | Whether a box crop is used is not stated. Aspect kept, height 32 |
| Genius-Society (potterhsu fork) | 89.21% | Code evaluates on val, so the split is unclear |
| thomalm | 85.05% | Use of extra not stated. 32x32 grayscale |
| carysva (same 33,402/13,068 split as ours) | 82.31% | EfficientNetB0 multi-head, 15% pad per side, 64x64 |
| shudhanshurp (Kaggle) | 76.33% CNN, 64.03% CRNN-CTC | Use of extra unknown, grayscale 64x64 |

No source runs a clean with/without-extra ablation. Comparing across repos, extra is worth about 5 to 14 points, but model differences confound that estimate.

### No GT crop (closest to our setup)

| Model | Seq. acc. | Caveat |
|---|---|---|
| He et al. 2019, whole image to 64x64 with random 54x54 crop, train+extra | 70.58% (CNN), 74.89% (soft attention), 77.61%, 79.04%, 80.45% (best) | The only clean whole-image benchmark. I checked these numbers against the paper text |
| pavitrakumar78, predicted box then classify | 51% | Two-stage pipeline, up to 4 digits |
| DRAM, CNN trained on tight crops and run on 110x110 looser crops | 50% (error) | Retrained at 110x110 the error is 5.60%. These looser crops are still centered on the box |

## 3. Why we get 74.7% val and 56.0% test, ranked

1. **No localization (whole image as input).** This is the main cause of the gap to 96%. He et al. lose 21.5 points with soft attention (96.47% to 74.89%), even with train+extra. Our val of 74.7% sits right at that uncropped level. Other numbers and text in the background explain extra predicted digits.
2. **Scale shift between train and test under the whole-image resize.** This most likely explains the 18.7-point val-test gap. Test images are larger (median width 132 vs 104 px), so after an aspect-kept resize the digits take up a different share of the canvas. DRAM shows a scale mismatch alone can push error to 50%. This is a hypothesis. Before acting on it, run the checks below.
3. **No extra set.** We train on about 30k images, while the 96% results use about 200k to 235k. From cross-repo comparison, this probably costs 5 to 14 points even after cropping.
4. **The left-to-right decoder over-predicts extra digits.** DRAM reports the same failure ("over predicts additional digits in the place of the terminal class"). That matches '9' -> '19' and '26' -> '260'. Cropping should remove much of it. What remains can be handled with length control.
5. **Minor recipe differences.** Teacher forcing decays to 0.5, label smoothing is 0.05, and the augmentation is a mild affine with no random-crop jitter. q19316 reaches about 96.2% with full teacher forcing and no label smoothing. None of these is likely to explain a 20 to 40 point gap.
6. **Checkpoint selection on val.** This inflates val a little. It cannot explain 18.7 points.

### Checks to run first (cheap, no retraining)

- Re-check test label parsing. digitStruct label 10 must map to digit 0, and digits must be ordered left to right. Compare per-length test accuracy with per-length val accuracy.
- Bin test accuracy by original image width and height. If accuracy falls off with size, reason 2 is confirmed.
- Run the current checkpoint on GT-cropped test images (same canvas). If accuracy jumps without retraining, localization is the main cause.

## 4. Changes to match common practice, with expected effect

| # | Change | Expected effect | Evidence |
|---|---|---|---|
| 1 | Crop train, val and test to the GT union box, expanded 1.3x around its center (0.15 x side per side), clipped to the image. Resize to 64x64 RGB (aspect not kept). Keep the ImageNet mean/std because the ResNet is pretrained | Largest gain. Expected test of about 82 to 91% with train only, and the val-test gap should shrink to a few points | carysva 82.31%, thomalm 85.05%, Genius 89.21%, Guo 91%. He et al. +21.5 points from cropping |
| 2 | Replace the mild affine with RandomCrop 54x54 at train and CenterCrop 54x54 at test (or RandomResizedCrop(54, scale 0.7-1.0) as in q19316). Keep the color jitter | About +0.5 points | Goodfellow Sec 5.1 |
| 3 | Add the extra set (202,353 images). Hold out a random 10% of train+extra as val, or keep val from train only so it stays comparable to the current number | About 95 to 96% test, based on the near-identical q19316 recipe. Costs about 7x the data per epoch | q19316 about 96.2%, SVHN_RNN 96.34%, potterhsu 95.65% |
| 4 | Make decoding length-aware. Options are an auxiliary length head on the pooled encoder features to constrain EOS, or a right-to-left decoder combined with the forward one | About +0.5 points at the 96% level. Could matter more for us because extra digits are our main error, but that is untested | DRAM single MC 4.4% vs forward-backward 3.9% error |
| 5 | Use full teacher forcing, as in q19316, or ablate the decay to 0.5 | Small and unmeasured. Simplifies the recipe | q19316 |
| 6 | Encoder geometry. With a 54x54 input, the current strides give a 2x7 token map, which is enough for 5 to 6 digits | Neutral | q19316 uses about 7x7 tokens |
| 7 | Report two results. (a) GT-crop protocol, comparable to the literature and labeled as oracle localization. (b) Whole-image result, compared with He et al. 70.58 to 80.45%. State whether the 6-digit train label was dropped | Makes the report honest and comparable | Genius-Society notes some courses forbid test boxes. Ours does not mention it |

## Caveats on sources

- Kaggle notebooks were not read directly. "Kaggle practice" here means GitHub/Kaggle repos plus the two Kaggle notebooks read through GitHub copies. Those two score lower (75.40%, 76.33%, 64.03%) than the Goodfellow reimplementations.
- hangyao's 93.7% is per-position accuracy, not sequence accuracy, so I left it out above.
- caoquanjie and Keras_SVHN selected checkpoints on test.
- potterhsu forks skip labels longer than 5.
- Goodfellow does not describe the test-time crop. The GT crop at test comes from the reimplementations.
- CTC does not appear among the high-scoring public implementations. The only CRNN-CTC result read on this protocol is 64.03% (shudhanshurp). Guo et al. report 91%, but their crop is not stated.
- Not verified and not cited: IEEE 8686963 (92.53%, seen only in a search snippet) and a ResearchGate encoder-decoder paper (96.57%, seen only in a snippet).
- Extracted paper text is in /tmp/claude-1000/-home-vungocduong-dl-final-prj/3c29501b-74ce-4dfe-b0a4-0fed46b25312/scratchpad/papers
- Course requirements checked in /home/vungocduong/dl_final_prj/Final Project Requirements 2026-1.pdf (Task 3 only requires a CNN encoder with an RNN/LSTM decoder, attention optional, and has no rule about bounding boxes)