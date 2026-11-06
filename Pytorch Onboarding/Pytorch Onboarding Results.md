# PyTorch Onboarding Results: Image Classification on CIFAR-10

This document compiles the training setup and results for three image classifiers built with PyTorch: an MLP, a basic CNN, and a deeper VGG-style CNN. It also includes an ablation study that measures what each of the deep CNN's original improvements contributes. Based on that study, Dropout was removed from the final deep CNN.

All numbers come from [`Pytorch_Onboarding.ipynb`](Pytorch_Onboarding.ipynb) on an NVIDIA RTX 4070:
- The three models (Sections 3–5 and 7) come from the latest complete run of the notebook.
- The ablation experiments (Section 6) come from an earlier run. Their per-epoch histories are saved in [`results/ablation_results.json`](results/ablation_results.json).

---

## 1. Task and dataset

- **Task:** classify 32×32 colour images into 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **Dataset:** CIFAR-10. 50,000 training images and 10,000 test images, evenly split across the 10 classes. Random guessing scores 10%.
- **Input pipeline:**
  - Images are converted to float tensors of shape `[3, 32, 32]` with values in [0, 1].
  - Each colour channel is normalized with the CIFAR-10 training-set statistics (mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)), so every channel is centered near 0 with roughly unit spread.
  - **Data augmentation** is used only by the deep CNN, and only on training images: a random 32×32 crop after 4-pixel padding (small random shifts), plus a random horizontal flip with probability 0.5. Test images are never augmented.

## 2. Common training setup

These settings are shared by every model, so differences in results come from the model or from the specific change being tested.

| Setting | Value |
|---|---|
| Loss | Cross-entropy (`nn.CrossEntropyLoss`). Models output raw logits with no softmax layer |
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 64 (training batches shuffled every epoch) |
| Random seed | 42, reset before each model so every run starts from the same initial weights |
| Evaluation | After every epoch, loss and accuracy on the full 10,000-image test set, in `model.eval()` mode (Dropout off, BatchNorm using its running statistics) |
| Hardware | NVIDIA RTX 4070, PyTorch 2.15 (CUDA 13.4) |

Each epoch follows the standard training loop for every batch: forward pass, compute the loss, zero the old gradients, backpropagate, then take an optimizer step.

---

## 3. Model 1: Multilayer Perceptron (MLP)

**Idea:** flatten each image into one 3,072-value vector and classify it with fully connected layers.

```
Input [3, 32, 32] → Flatten → 3072
→ Linear(3072, 512) → ReLU
→ Linear(512, 256)  → ReLU
→ Linear(256, 10)   → logits
```

- **Parameters:** 1,707,274. About 92% of them (1.57M) are in the first layer, because every input pixel connects to every hidden unit.
- **Training:** 10 epochs, constant learning rate, no augmentation.

**Results**

| Best test acc | Final test acc | Final train acc | Lowest test loss | Final test loss | Training time |
|---|---|---|---|---|---|
| 53.11% (epoch 7) | 52.91% | 67.68% | 1.385 (epoch 4) | 1.522 | 40 s |

**Observations**
- **Accuracy levels off near 53%.** Test accuracy stops improving after about epoch 4, while training accuracy keeps rising to 67.7%.
- **Overfitting starts by epoch 4.** Test loss is lowest at epoch 4 and climbs afterwards. The model is memorizing the training set instead of learning features that generalize.
- **Why an MLP struggles on images.** Flattening discards spatial structure: the network has no built-in idea of which pixels are next to each other. A pattern learned in one part of the image also doesn't carry over to other positions.

---

## 4. Model 2: Basic CNN

**Idea:** convolution layers slide small learned 3×3 filters across the image to detect local patterns such as edges and textures. The same filter is reused at every position (*weight sharing*), so a conv layer has very few parameters and recognizes a pattern wherever it appears (*translation equivariance*). Max-pooling halves the spatial size, which makes the features more robust to small shifts and lets later layers see larger regions.

```
Input                                   [3, 32, 32]
Conv(3→32, 3×3, pad 1)  → ReLU → MaxPool(2)   → [32, 16, 16]
Conv(32→64, 3×3, pad 1) → ReLU → MaxPool(2)   → [64, 8, 8]
Flatten                                       → 4096
Linear(4096, 256) → ReLU → Linear(256, 10)
```

- **Parameters:** 1,070,794, which is 37% fewer than the MLP. The two conv layers hold only 19,392 of them. Nearly all the rest are in the first fully connected layer.
- **Training:** the same as the MLP: 10 epochs, constant learning rate, no augmentation.

**Results**

| Best test acc | Final test acc | Final train acc | Lowest test loss | Final test loss | Training time |
|---|---|---|---|---|---|
| 73.64% (epoch 6) | 71.51% | 96.83% | 0.808 (epoch 4) | 1.519 | 41 s |

**Observations**
- **+20.5 points over the MLP with fewer parameters.** Exploiting spatial structure matters far more than raw parameter count.
- **Overfitting is severe.** Training accuracy reaches 96.8% while test accuracy levels off around 72–73%, a 25.3-point gap at the final epoch. Test loss is lowest at epoch 4 and then nearly doubles, from 0.81 to 1.52. The network has the capacity to memorize the training set, and nothing in the setup stops it.
- **This motivates the deep CNN.** It needs more capacity, to learn richer features, and regularization, to stop the memorization.

---

## 5. Model 3: Deep CNN (final model)

**Idea:** a VGG-style network. Three blocks each double the channel count while halving the spatial size. Four additions over the basic CNN make the deeper network train well and generalize:

1. **Depth:** 6 conv layers instead of 2. Two stacked 3×3 convs per block see a 5×5 region with fewer weights than one 5×5 conv, and the extra ReLU between them lets the block learn more complex features.
2. **Batch Normalization** after every conv. It normalizes each channel over the batch, which keeps activations well scaled through the deeper network and makes optimization faster and more stable. The conv layers drop their bias, because BatchNorm's learned shift makes it redundant.
3. **Data augmentation:** random crop and horizontal flip (Section 1). The network effectively never sees the same training image twice.
4. **Cosine-annealing learning-rate schedule** over **30 epochs**, three times as many as Models 1–2. The learning rate decays smoothly from 1e-3 toward 0: large steps early make fast progress, and small steps late let the model settle into a good minimum.

**No Dropout in the final model.** The original design also had Dropout (p = 0.5) before each fully connected layer. The ablation study (Section 6) showed it gave no measurable benefit once BatchNorm and augmentation were in place: the variant without it scored slightly higher (+0.32 points). Dropout was therefore removed, and the final model was retrained without it.

```
conv_block(c_in, c_out) = [Conv3×3 → BatchNorm → ReLU] × 2 → MaxPool(2)

Input                         [3, 32, 32]
conv_block(3 → 64)          → [64, 16, 16]
conv_block(64 → 128)        → [128, 8, 8]
conv_block(128 → 256)       → [256, 4, 4]
Flatten                     → 4096
Linear(4096, 512) → ReLU → Linear(512, 10)
```

- **Parameters:** 3,249,098. That's 1,146,304 in the conv and BatchNorm layers and 2,102,794 in the classifier. Dropout has no parameters, so removing it doesn't change the count.

**Results**

| Best test acc | Final test acc | Final train acc | Lowest test loss | Final test loss | Training time |
|---|---|---|---|---|---|
| 90.71% (epoch 30) | 90.71% | 97.82% | 0.348 (epoch 24) | 0.354 | 181 s (6.0 s/epoch) |

For comparison, the original version with Dropout reached 90.29% best and 90.23% final test accuracy, with 94.38% train accuracy (Section 6, "Original deep CNN").

**Observations**
- **+17.1 points over the basic CNN and +37.6 over the MLP.** This is the best of the three models.
- **Overfitting is under control.**
  - The final train−test gap is 7.1 points, compared with 25.3 for the basic CNN.
  - That's wider than the 4.2-point gap of the version with Dropout, because the model now fits the training data more closely. Test accuracy didn't suffer: it is slightly higher.
  - Test loss falls until epoch 24 and then stays essentially flat (0.348 → 0.354), instead of climbing as it did for the basic CNN.
  - Test accuracy reaches its best value at the final epoch, so the best and final numbers are the same.
- **Early on, test accuracy is higher than training accuracy (epochs 1–6).** That's expected: training accuracy is measured on augmented images, which are harder than the clean test images.

---

## 6. Experiments: ablation study of the deep CNN

The original deep CNN changed six things at once compared with the basic CNN: depth, BatchNorm, Dropout, augmentation, LR schedule, and training length. To measure what each contributes, the **original deep CNN (with Dropout) was retrained with exactly one improvement removed** at a time (a leave-one-out ablation). All other settings and the random seed were kept identical. The bigger the drop in accuracy, the more that component matters.

The final model in Section 5 is the "No Dropout" configuration from this table, retrained in the final notebook cell.

| Variant | What changed | Params | Epochs | Best test acc | vs. original | Final train acc | Final test acc | Train−test gap | Time |
|---|---|---|---|---|---|---|---|---|---|
| No Dropout (→ final model) | Dropout p = 0 | 3,249,098 | 30 | **90.61%** | +0.32 | 97.78% | 90.61% | +7.2 | 199 s |
| **Original deep CNN** | none (all six improvements, incl. Dropout 0.5) | 3,249,098 | 30 | **90.29%** | 0 | 94.38% | 90.23% | +4.2 | 204 s |
| No LR schedule | constant lr = 1e-3 | 3,249,098 | 30 | 89.48% | −0.81 | 93.04% | 89.12% | +3.9 | 197 s |
| No augmentation | plain training images | 3,249,098 | 30 | 88.43% | −1.86 | 99.65% | 88.33% | +11.3 | 163 s |
| Short training | 10 epochs (cosine over 10) | 3,249,098 | 10 | 85.23% | −5.06 | 84.74% | 85.23% | −0.5 | 65 s |
| Shallower | 1 conv per block (3 conv layers) | 2,474,058 | 30 | 84.49% | −5.80 | 82.40% | 84.36% | −2.0 | 198 s |
| No BatchNorm | BatchNorm removed (conv bias restored) | 3,248,202 | 30 | 83.89% | −6.40 | 85.83% | 83.63% | +2.2 | 195 s |
| *Basic CNN (reference)* | Model 2 | 1,070,794 | 10 | 73.64% | −16.65 | 96.83% | 71.51% | +25.3 | 41 s |

*"vs. original" is the difference in best test accuracy from the original deep CNN, in percentage points. "Train−test gap" is final train accuracy minus final test accuracy, in percentage points; a large positive gap means overfitting. The Basic CNN reference row comes from the latest run (Section 4).*

### Findings, ranked by impact

1. **BatchNorm had the largest effect (−6.4 points).**
   - Without it, the 6-layer network trains much more slowly: it reaches only 85.8% *training* accuracy after 30 epochs, compared with 94.4% with BatchNorm.
   - This is **underfitting, not overfitting**. BatchNorm's main benefit here is making the deeper network easy to optimize.
2. **Depth (−5.8 points).**
   - With one conv per block, the network ends with training accuracy (82.4%) *below* test accuracy (84.4%). It doesn't have enough capacity to fit even the augmented training data.
   - The second conv in each block adds only about 0.78M parameters but is worth almost 6 points.
3. **Training length (−5.1 points).**
   - At 10 epochs the model is still clearly improving: its train and test accuracy are nearly equal and both still rising.
   - The regularized model needs, and benefits from, longer training. In contrast, the unregularized basic CNN had stopped improving by about epoch 4–6.
   - **At the same 10-epoch budget, the deep CNN still beats the basic CNN by 11.6 points** (85.23% vs. 73.64%). So the improvement isn't only due to longer training.
4. **Data augmentation (−1.9 points in accuracy, with the clearest effect on overfitting).**
   - Without augmentation, training accuracy reaches 99.65%: the network memorizes the training set. The train−test gap nearly triples, from 4.2 to 11.3 points.
   - Test loss is lowest at epoch 13 (0.452) and then rises to 0.671 by epoch 30, double the original model's final test loss of 0.340.
   - Accuracy holds up better than loss because BatchNorm, Dropout, and the decaying learning rate still provide some regularization. But the model becomes much more overconfident on the examples it gets wrong.
5. **LR schedule (−0.8 points).**
   - Without cosine annealing, test accuracy fluctuates from epoch to epoch late in training. For example, it ranges from 88.0% to 89.5% over the last 10 epochs, compared with a smooth climb to 90.2% with the schedule.
   - Decaying the learning rate lets the optimizer settle instead of bouncing around a minimum.
6. **Dropout (+0.3 points, so removing it slightly *helped* accuracy). Dropout was removed from the final model.**
   - Without Dropout, the model fits the training data much more closely (97.8% vs. 94.4% train accuracy, with a gap of 7.2 vs. 4.2 points), but test accuracy is about the same.
   - With BatchNorm and data augmentation already regularizing the convolutional layers, Dropout in the classifier adds little at 30 epochs. It does reduce overfitting, but that doesn't turn into better test accuracy here.
   - A 0.3-point difference is within the run-to-run variation expected from a single seed, so this result should be read as "no measurable benefit", not "Dropout hurts". Since it doesn't help, the simpler model without it was kept.
   - **Confirmed by the retrain:** the final model, with the same configuration as "No Dropout", reached 90.71%. That's within 0.1 points of the ablation run and again above the original's 90.29%.

### Summary

The components fall into two groups:
- **Components that help the network learn:** BatchNorm, depth, and enough training epochs. Each is worth 5–6 points. Without them the model *underfits*.
- **Components that limit overfitting:** augmentation, Dropout, and the LR schedule. They matter most for the gap between training and test performance and for test loss. Their effect on peak accuracy is smaller, because the other regularizers partly compensate when one is removed. Dropout turned out to be redundant with the others, so the final model keeps only augmentation and the LR schedule.

---

## 7. Overall comparison of the three models

| Model | Params | Epochs | Best test acc | Final test acc | Final train acc | Train−test gap | Training time |
|---|---|---|---|---|---|---|---|
| MLP | 1,707,274 | 10 | 53.11% | 52.91% | 67.68% | +14.8 | 40 s |
| Basic CNN | 1,070,794 | 10 | 73.64% | 71.51% | 96.83% | +25.3 | 41 s |
| Deep CNN (final, no Dropout) | 3,249,098 | 30 | **90.71%** | **90.71%** | 97.82% | +7.1 | 181 s |
---

## 8. Figures

Training curves are in [`OutputGraphs/CurrRuns/`](OutputGraphs/CurrRuns/):

| Figure | Contents |
|---|---|
| [`MLPoutput.png`](OutputGraphs/CurrRuns/MLPoutput.png) | MLP training and test loss and accuracy |
| [`CNNoutput.png`](OutputGraphs/CurrRuns/CNNoutput.png) | Basic CNN training and test loss and accuracy |
| [`MLPvsCNNoutput.png`](OutputGraphs/CurrRuns/MLPvsCNNoutput.png) | MLP and basic CNN on shared axes |
| [`DeepCNNoutput.png`](OutputGraphs/CurrRuns/DeepCNNoutput.png) | Final deep CNN (no Dropout) training and test loss and accuracy |
| [`AllModelsVSoutput.png`](OutputGraphs/CurrRuns/AllModelsVSoutput.png) | All three models on shared axes |
| [`CNNExperimentoutput.png`](OutputGraphs/CurrRuns/CNNExperimentoutput.png) | Ablation study: test accuracy per epoch, and best test accuracy, for every variant |

`MLPoutput.png`, `CNNoutput.png` and `MLPvsCNNoutput.png` come from the previous run of the same code. The basic CNN's curves there differ slightly from the numbers above (72.61% best test accuracy in that run).

Figures from the earlier Colab runs are kept in [`OutputGraphs/OldRuns/`](OutputGraphs/OldRuns/).
