# ECE531-Final-Project

# ECE531 Term Project: Frequency-Enhanced Camouflaged Object Segmentation

This repository contains the implementation notebook and result files for the ECE531 term project:

**Frequency-Enhanced Camouflaged Object Segmentation Using FFT-Based Spatial-Frequency Fusion**

The project investigates whether FFT-based frequency information can improve camouflaged object segmentation when used as an additional input cue or as a frequency-guided modulation signal.

## Repository Contents

├── ece531-term-project.ipynb
├── README.md
├── requirements.txt

## Implementation Note

The experiments reported in the paper were conducted using the same notebook-based training and evaluation pipeline. During experimentation, the relevant cells/configuration settings were modified to test different model variants, loss functions, frequency maps, resolutions, and augmentation strategies.

The final notebook includes the main implementation pipeline and the two most important model variants:

1. **Input-level FFT Fusion**: ResNet18-UNet with RGB + FFT high-frequency map as a 4-channel input.
2. **Frequency-Guided Spatial-Frequency Fusion**: ResNet18-UNet variant where the FFT map is used as an attention/modulation signal over the RGB input representation.

The complete numerical results from the ablation study are provided in `ablation_results.csv`.

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

## Dataset

The datasets are not included in this repository due to size limitations. The experiments use CAMO and COD10K datasets with the following expected structure:

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

Non-camouflaged COD10K samples with empty masks were excluded, following the setup described in the paper.

## Main Results

The two final candidate models are:

| Model                   | CAMO Val Dice | COD10K Test Dice | Notes                                                           |
| ----------------------- | ------------: | ---------------: | --------------------------------------------------------------- |
| Input-level FFT Fusion  |        0.6907 |           0.6470 | Best CAMO validation performance and strongest simple baseline. |
| Frequency-Guided Fusion |        0.6591 |           0.6495 | Best held-out COD10K test performance and selected final model. |

The Frequency-Guided Fusion model is selected as the primary final model because the COD10K test set is treated as the stronger indicator of cross-dataset generalisation.

## Environment

The notebook was developed and tested in the Kaggle GPU environment.

Install the main dependencies with:

pip install -r requirements.txt

## Notes

Trained model checkpoints are not included because of file size limitations. The notebook saves checkpoints and prediction outputs under the working/output directory when training is run.

The reported results may vary slightly across runs due to random initialisation, data augmentation, and GPU nondeterminism.
