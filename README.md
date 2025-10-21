# Whale-VAD: Whale Vocalisation Activity Detection

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17251589-blue)](https://doi.org/10.5281/zenodo.17251589)
[![Paper](https://img.shields.io/badge/Paper-DCASE%202025-green)](https://doi.org/10.5281/zenodo.17251589)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)

<!-- Banner placeholder -->
<!--![Whale-VAD Banner](assets/banner.png)-->

A lightweight sound event detection system for discovering whale calls in marine audio recordings. This repository contains the implementation of our hybrid CNN-BiLSTM architecture with residual bottleneck and depthwise convolutions, designed for coherent per-frame whale call event detection.

<!-- DCASE/BioDCASE logo placeholder -->
<p align="center">
  <img src="assets/dcase_logo.png" alt="DCASE Logo" width="200"/>
  <img src="assets/biodcase_logo.png" alt="BioDCASE Logo" width="200"/>
</p>

## Getting Started

Whale-VAD uses PyTorch Hub for easy model loading and inference. The model automatically handles feature extraction and produces frame-level probability outputs for three whale call types: `bmabz`, `d`, and `bp`.

```python
import torch
import torchaudio as ta

# Load the model
classifier, transform = torch.hub.load("CMGeldenhuys/Whale-VAD", 'whalevad', weights='DEFAULT')

# Load audio file (must be sampled at 250 Hz, single channel)
audio, sr = ta.load("whale-call.wav")
assert sr == 250

# Perform inference
features = transform(audio)
prob = classifier(features)  # Frame-level probabilities for bmabz, d, and bp
```

### Requirements

- Python >= 3.11.13
- PyTorch >= 2.7.1
- torchaudio

### Input Specifications

- **Sample Rate**: 250 Hz (required)
- **Channels**: Single channel (mono) audio
- **Format**: Any format supported by torchaudio

### Output

The model produces frame-level probability outputs for three whale call types:
- `bmabz`: Blue whale calls (BmA, BmB, BmZ)
- `d`: D-calls (BmD and BpD)
- `bp`: Fin whale calls (Bp20 and Bp20plus)

## Dataset

This model was trained on the Acoustic Trends Blue Fin Library (ATBFL) dataset as part of the BioDCASE 2025 Challenge (Task 2).

- **Challenge Website**: [https://biodcase.github.io/challenge2025/task2](https://biodcase.github.io/challenge2025/task2)
- **Dataset DOI**: [https://doi.org/10.5281/zenodo.15092732](https://doi.org/10.5281/zenodo.15092732)

## Model Weights

Pre-trained model weights are available in the [GitHub Releases](https://github.com/CMGeldenhuys/Whale-VAD/releases) section. Weights can be loaded automatically via PyTorch Hub or downloaded manually.

## Citation

If you use this work in your research, please cite:

```
Geldenhuys, C., Tonitz, G., & Niesler, T. (2025). Whale-VAD: Whale Vocalisation Activity Detection. Proceedings of the 10th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2025), 165–169. https://doi.org/10.5281/zenodo.17251589
```

```bibtex
@inproceedings{Geldenhuys2025,
    author = "Geldenhuys, Christiaan and Tonitz, Günther and Niesler, Thomas",
    title = "Whale-VAD: Whale Vocalisation Activity Detection",
    booktitle = "Proceedings of the 10th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2025)",
    address = "Barcelona, Spain",
    month = "October",
    year = "2025",
    pages = "165--169",
    isbn = "978-84-09-77652-8",
    doi = "10.5281/zenodo.17251589"
}
```

## Contributing

We welcome contributions to improve Whale-VAD! Please feel free to submit issues, fork the repository, and create pull requests.


## Authors

- **Christiaan M. Geldenhuys** [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1234--5678-green.svg)](https://orcid.org/0000-0002-1234-5678)

- **Günther Tonitz** [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1234--5678-green.svg)](https://orcid.org/0000-0002-1234-5678)

- **Thomas R. Niesler** [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1234--5678-green.svg)](https://orcid.org/0000-0002-1234-5678)

## Acknowledgements

The authors gratefully acknowledge Telkom (South Africa) for their financial support, and the [Stellenbosch Rhasatsha high performance computing (HPC)](https://www0.sun.ac.za/hpc) facility for the compute time provided to the research presented in this work.

<p align="center">
    <img src="assets/Telkom_logo.png" alt="Telkom  Logo" width="200"/>
    <img src="assets/SU_logo.png" alt="Stellenbosch University Logo" width="200"/>
</p>

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - a copyleft license that requires anyone who distributes the code or a derivative work to make the source available under the same terms. All code and model weights are provided as is.

---

*Presented at the 10th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2025), Barcelona, Spain, October 2025.*
