# Image Classification with PyTorch on CIFAR-10: Report Notes

Source material for the report: what each model is, how it was trained, and the measured results.
All numbers come from one complete run of `Pytorch_Onboarding.ipynb` on a local NVIDIA RTX 4070.
Raw per-epoch histories for the deep CNN experiments are saved in `results/ablation_results.json`.

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
| 53.11% (epoch 7) | 52.91% | 67.68% | 1.385 (epoch 4) | 1.522 | 45 s |

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
| 72.61% (epoch 7) | 71.18% | 96.82% | 0.826 (epochs 3–4) | 1.492 | 44 s |

**Observations**
- **+19.5 points over the MLP with fewer parameters.** Exploiting spatial structure matters far more than raw parameter count.
- **Overfitting is severe.** Training accuracy reaches 96.8% while test accuracy levels off around 72%, a 25.6-point gap. Test loss is lowest at epochs 3–4 and then nearly doubles, from 0.83 to 1.49. The network has the capacity to memorize the training set, and nothing in the setup stops it.
- **This motivates the deep CNN.** It needs more capacity, to learn richer features, and regularization, to stop the memorization.

---

## 5. Model 3: Deep CNN (final model)

**Idea:** a VGG-style network. Three blocks each double the channel count while halving the spatial size. Five additions over the basic CNN make the deeper network train well and generalize:

1. **Depth:** 6 conv layers instead of 2. Two stacked 3×3 convs per block see a 5×5 region with fewer weights than one 5×5 conv, and the extra ReLU between them lets the block learn more complex features.
2. **Batch Normalization** after every conv. It normalizes each channel over the batch, which keeps activations well scaled through the deeper network and makes optimization faster and more stable. The conv layers drop their bias, because BatchNorm's learned shift makes it redundant.
3. **Dropout (p = 0.5)** before each fully connected layer. It randomly zeroes activations during training so the classifier can't depend on any single feature. It is switched off at test time.
4. **Data augmentation:** random crop and horizontal flip (Section 1). The network effectively never sees the same training image twice.
5. **Cosine-annealing learning-rate schedule** over **30 epochs**, three times as many as Models 1–2. The learning rate decays smoothly from 1e-3 toward 0: large steps early make fast progress, and small steps late let the model settle into a good minimum.

```
conv_block(c_in, c_out) = [Conv3×3 → BatchNorm → ReLU] × 2 → MaxPool(2)

Input                         [3, 32, 32]
conv_block(3 → 64)          → [64, 16, 16]
conv_block(64 → 128)        → [128, 8, 8]
conv_block(128 → 256)       → [256, 4, 4]
Flatten                     → 4096
Dropout(0.5) → Linear(4096, 512) → ReLU → Dropout(0.5) → Linear(512, 10)
```

- **Parameters:** 3,249,098. That's 1,146,304 in the conv and BatchNorm layers and 2,102,794 in the classifier.

**Results**

| Best test acc | Final test acc | Final train acc | Lowest test loss | Final test loss | Training time |
|---|---|---|---|---|---|
| 90.29% (epoch 29) | 90.23% | 94.38% | 0.328 (epoch 24) | 0.340 | 204 s (6.8 s/epoch) |

**Observations**
- **+17.7 points over the basic CNN and +37.2 over the MLP.** This is the best model in the project.
- **Overfitting is under control.**
  - The final train−test gap is only 4.2 points, compared with 25.6 for the basic CNN.
  - Test loss keeps falling almost all the way to the end, reaching its lowest value at epoch 24.
  - Test accuracy is still rising slowly at epoch 30.
- **Early on, test accuracy is higher than training accuracy.** That's expected: training accuracy is measured on augmented images with Dropout active, which makes training harder than testing.

---

## 6. Experiments: ablation study of the deep CNN

The deep CNN changes six things at once compared with the basic CNN: depth, BatchNorm, Dropout, augmentation, LR schedule, and training length. To measure what each contributes, the **full deep CNN was retrained with exactly one improvement removed** at a time (a leave-one-out ablation). All other settings and the random seed were kept identical. The bigger the drop in accuracy, the more that component matters.

| Variant | What changed | Params | Epochs | Best test acc | vs. full | Final train acc | Final test acc | Train−test gap | Time |
|---|---|---|---|---|---|---|---|---|---|
| No Dropout | Dropout p = 0 | 3,249,098 | 30 | **90.61%** | +0.32 | 97.78% | 90.61% | +7.2 | 199 s |
| **Full Deep CNN** | none | 3,249,098 | 30 | **90.29%** | 0 | 94.38% | 90.23% | +4.2 | 204 s |
| No LR schedule | constant lr = 1e-3 | 3,249,098 | 30 | 89.48% | −0.81 | 93.04% | 89.12% | +3.9 | 197 s |
| No augmentation | plain training images | 3,249,098 | 30 | 88.43% | −1.86 | 99.65% | 88.33% | +11.3 | 163 s |
| Short training | 10 epochs (cosine over 10) | 3,249,098 | 10 | 85.23% | −5.06 | 84.74% | 85.23% | −0.5 | 65 s |
| Shallower | 1 conv per block (3 conv layers) | 2,474,058 | 30 | 84.49% | −5.80 | 82.40% | 84.36% | −2.0 | 198 s |
| No BatchNorm | BatchNorm removed (conv bias restored) | 3,248,202 | 30 | 83.89% | −6.40 | 85.83% | 83.63% | +2.2 | 195 s |
| *Basic CNN (reference)* | Model 2 | 1,070,794 | 10 | 72.61% | −17.68 | 96.82% | 71.18% | +25.6 | 44 s |

*"vs. full" is the difference in best test accuracy, in percentage points. "Train−test gap" is final train accuracy minus final test accuracy, in percentage points; a large positive gap means overfitting.*

### Findings, ranked by impact

1. **BatchNorm had the largest effect (−6.4 points).**
   - Without it, the 6-layer network trains much more slowly: it reaches only 85.8% *training* accuracy after 30 epochs, compared with 94.4% with BatchNorm.
   - This is **underfitting, not overfitting**. BatchNorm's main benefit here is making the deeper network easy to optimize.
2. **Depth (−5.8 points).**
   - With one conv per block, the network ends with training accuracy (82.4%) *below* test accuracy (84.4%). It doesn't have enough capacity to fit even the augmented training data.
   - The second conv in each block adds only about 0.78M parameters but is worth almost 6 points.
3. **Training length (−5.1 points).**
   - At 10 epochs the full model is still clearly improving: its train and test accuracy are nearly equal and both still rising.
   - The regularized model needs, and benefits from, longer training. In contrast, the unregularized basic CNN had stopped improving by about epoch 4.
   - **At the same 10-epoch budget, the deep CNN still beats the basic CNN by 12.6 points** (85.23% vs. 72.61%). So the improvement isn't only due to longer training.
4. **Data augmentation (−1.9 points in accuracy, with the clearest effect on overfitting).**
   - Without augmentation, training accuracy reaches 99.65%: the network memorizes the training set. The train−test gap nearly triples, from 4.2 to 11.3 points.
   - Test loss is lowest at epoch 13 (0.452) and then rises to 0.671 by epoch 30, double the full model's final test loss of 0.340.
   - Accuracy holds up better than loss because BatchNorm, Dropout, and the decaying learning rate still provide some regularization. But the model becomes much more overconfident on the examples it gets wrong.
5. **LR schedule (−0.8 points).**
   - Without cosine annealing, test accuracy fluctuates from epoch to epoch late in training. For example, it ranges from 88.0% to 89.5% over the last 10 epochs, compared with a smooth climb to 90.2% with the schedule.
   - Decaying the learning rate lets the optimizer settle instead of bouncing around a minimum.
6. **Dropout (+0.3 points, so removing it slightly *helped* accuracy).**
   - Without Dropout, the model fits the training data much more closely (97.8% vs. 94.4% train accuracy, with a gap of 7.2 vs. 4.2 points), but test accuracy is about the same.
   - With BatchNorm and data augmentation already regularizing the convolutional layers, Dropout in the classifier adds little at 30 epochs. It does reduce overfitting, but that doesn't turn into better test accuracy here.
   - A 0.3-point difference is within the run-to-run variation expected from a single seed, so this result should be read as "no measurable benefit", not "Dropout hurts".

### Summary

The components fall into two groups:
- **Components that help the network learn:** BatchNorm, depth, and enough training epochs. Each is worth 5–6 points. Without them the model *underfits*.
- **Components that limit overfitting:** augmentation, Dropout, and the LR schedule. They matter most for the gap between training and test performance and for test loss. Their effect on peak accuracy is smaller, because the other regularizers partly compensate when one is removed.

---

## 7. Overall comparison of the three models

| Model | Params | Epochs | Best test acc | Final test acc | Final train acc | Train−test gap | Training time |
|---|---|---|---|---|---|---|---|
| MLP | 1,707,274 | 10 | 53.11% | 52.91% | 67.68% | +14.8 | 45 s |
| Basic CNN | 1,070,794 | 10 | 72.61% | 71.18% | 96.82% | +25.6 | 44 s |
| Deep CNN | 3,249,098 | 30 | **90.29%** | **90.23%** | 94.38% | +4.2 | 204 s |

**Key takeaways**
- **Architecture matters more than size.** The basic CNN has fewer parameters than the MLP but scores 19.5 points higher, because convolutions build in useful assumptions about images: local patterns, and the same pattern can appear anywhere.
- **Capacity without regularization overfits.** Both 10-epoch models stop improving after 4–7 epochs, and their test loss rises while training accuracy keeps climbing.
- **The deep CNN reached about 90%** by combining more capacity (depth with BatchNorm) with regularization (augmentation and Dropout) and a decaying learning rate. Those additions also let it benefit from three times as many epochs without overfitting.

---

## 8. Limitations and notes

- **The "best test accuracy" numbers are slightly optimistic.** They are the maximum over epochs of the test-set accuracy, which means the best epoch was chosen using the test set. A rigorous setup would hold out a validation split, for example 5,000 training images, for that choice. The final-epoch test accuracy is reported alongside as an unbiased number, and for the deep CNN the two differ by less than 0.1 points.
- **Every configuration was run once, with seed 42.** GPU training isn't bit-for-bit deterministic, so differences under about 0.5 points between ablation variants, such as Dropout's +0.32, aren't conclusive. Repeating each variant with 3 seeds and reporting mean ± std would make the ablation stronger.
- **Earlier Colab runs gave slightly different numbers.** The same models trained on a Colab T4 reached 53.0% (MLP), 73.5% (basic CNN), and 90.07% (deep CNN), consistent with the run-to-run variation above. All numbers in this document come from the local RTX 4070 run.
- **Training times were measured on an RTX 4070.** The MLP and basic CNN take about the same time because data loading, not computation, limits them.

## 9. Figures

The saved figures in `OutputGraphs/` are from the **earlier Colab runs** and differ slightly from the tables above. Re-save them from the local run, or label them as Colab results in the report.

- `OutputGraphs/MLPCUDAoutput2.png`: MLP training and test loss and accuracy curves.
- `OutputGraphs/BasicCNNoutput.png`: basic CNN curves.
- `OutputGraphs/BetterCNNoutput.png`: deep CNN curves.
- `OutputGraphs/Finaloutput.png`: all three models on shared axes.
- *Not yet saved:* the ablation figure (test accuracy per epoch for every variant, and best test accuracy per variant) produced by the experiments cell.
