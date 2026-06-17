# ECE531 Term Project: Frequency-Enhanced Camouflaged Object Segmentation

This repository contains the implementation notebook and result files for the ECE531 term project:

**Frequency-Enhanced Camouflaged Object Segmentation Using FFT-Based Spatial-Frequency Fusion**

The project investigates whether FFT-based frequency information can improve camouflaged object segmentation when used as an additional input cue or as a frequency-guided modulation signal.

## Repository Contents

```
├── ece531-term-project.ipynb
├── README.md
└── requirements.txt
```

## Implementation Note

The experiments reported in the paper were conducted using the same notebook-based training and evaluation pipeline. During experimentation, the relevant cells/configuration settings were modified to test different model variants, loss functions, frequency maps, resolutions, and augmentation strategies.

The final notebook includes the main implementation pipeline and the two most important model variants:

1. **Input-level FFT Fusion**: ResNet18-UNet with RGB + FFT high-frequency map as a 4-channel input.
2. **Frequency-Guided Spatial-Frequency Fusion**: ResNet18-UNet variant where the FFT map is used as an attention/modulation signal over the RGB input representation.

The complete numerical results of the 13-condition ablation study are reported in Table I of the paper.

## Environment

Developed and tested in the Kaggle GPU environment (Tesla T4). Key dependencies:

- torch 2.10.0, torchvision 0.25.0
- numpy 2.0.2, pandas 2.3.3
- opencv-python 4.13.0, albumentations 1.4.0
- scikit-image, tqdm, matplotlib
- segmentation-models-pytorch (for the U-Net++ decoder variant)
- CUDA 12.8

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The datasets are not included due to size limitations. Download the **CAMO** and **COD10K** datasets and arrange them as follows:

```
data/
├── CAMO/
│   ├── train/images/
│   ├── train/masks/
│   ├── test/images/
│   └── test/masks/
└── COD10K/
    ├── train/images/
    ├── train/masks/
    ├── test/images/
    └── test/masks/
```

Non-camouflaged COD10K samples with empty ground-truth masks are excluded, following the setup described in the paper. After exclusion: CAMO has 1,000 train / 250 test (the test split is used as the validation set), and COD10K has 3,040 train / 2,026 test (held out strictly).

## Reproduction (Training, Inference, Evaluation)

The full pipeline is contained in `ece531-term-project.ipynb`. Run the cells top to bottom:

1. **Setup & data:** install/import dependencies, build the FFT frequency maps, and construct the 4-channel datasets and loaders.
2. **Input-level FFT model — Exp. 4:** train the ResNet18-UNet, select the segmentation threshold on the CAMO validation set (grid 0.30–0.90, step 0.05), and evaluate pixel-level and COD-standard metrics.
3. **Frequency-Guided Fusion model — Exp. 13:** train and evaluate the attention-modulation variant, then produce the validation-based final-model-selection table.
4. **Failure analysis:** rank COD10K test images by per-image Dice and generate the hardest-case figure.

The remaining 11 ablation conditions (Table I in the paper) are reproduced by editing the configuration as described in the **Ablation Experiments** table below (loss function, frequency map, backbone, resolution, augmentation, decoder).

## Ablation Experiments

The following table explains how each experiment in the paper was produced from the same notebook pipeline.

| Exp. | Experiment                      | Main Change                                                                                    |
| ---- | ------------------------------- | ---------------------------------------------------------------------------------------------- |
| 1    | Focal-Dice-Sobel                | Used RGB+FFT input with Focal+Dice+Sobel boundary loss.                                        |
| 2    | ResNet34 backbone               | Replaced the ResNet18 encoder with ResNet34.                                                   |
| 3    | RGB-only baseline               | Disabled the frequency information by using RGB input with a zero frequency channel.           |
| 4    | Input-level FFT Fusion          | Used RGB + FFT high-frequency map as a 4-channel input with Focal+Dice loss.                   |
| 5    | Boundary-band loss              | Added boundary-band Dice loss around the ground-truth contour.                                 |
| 6    | 384 px, batch size 6            | Increased input resolution to 384×384 and reduced batch size to 6 due to GPU memory limits.    |
| 7    | 384 px, batch size 8 + ReduceLR | Used 384×384 input with batch size 8 and ReduceLROnPlateau scheduling.                         |
| 8    | DoG frequency                   | Replaced the FFT frequency map with a Difference-of-Gaussians frequency map.                   |
| 9    | LoG frequency                   | Replaced the FFT frequency map with a Laplacian-of-Gaussian frequency map.                     |
| 10   | Object-centred crop             | Enabled object-centred crop augmentation.                                                      |
| 11   | U-Net++ decoder                 | Replaced the U-Net decoder with an SMP U-Net++ decoder using a ResNet18 encoder.               |
| 12   | Tversky loss                    | Added Tversky loss with α = 0.3 and β = 0.7.                                                   |
| 13   | Frequency-Guided Fusion         | Used the FFT map to generate an attention/modulation signal over the RGB input representation. |

## Main Results

Model selection is performed **exclusively on the CAMO validation set**; the COD10K test set is held out strictly and is never used for model selection.

| Model                             | CAMO Val Dice | COD10K Test Dice | Notes                                                               |
| --------------------------------- | ------------: | ---------------: | ------------------------------------------------------------------- |
| Input-level FFT Fusion (Exp. 4)   |        0.6907 |           0.6470 | Best CAMO validation performance — **selected primary model**.      |
| Frequency-Guided Fusion (Exp. 13) |        0.6591 |           0.6495 | Best held-out COD10K Dice; reported as an observation, not selected. |

Under the validation-only protocol, **Input-level FFT Fusion (Exp. 4)** is the selected primary model. The Frequency-Guided Fusion variant attains a marginally higher COD10K Dice (0.6495 vs. 0.6470), but since COD10K is held out, this is reported as an observation rather than used as a selection criterion.

## Notes

Trained model checkpoints are not included due to file size limitations. The notebook saves checkpoints and prediction outputs under the working/output directory when training is run.

Reported results may vary slightly across runs due to random initialization, data augmentation, and GPU non-determinism. For closer reproduction, set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
