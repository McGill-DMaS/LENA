# LENA: Llama-based Embeddings of Neutralized Assembly

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TSE-00629B.svg)](https://doi.org/10.1109/TSE.2026.3705321)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTSE.2026.3705321-007EC6.svg)](https://doi.org/10.1109/TSE.2026.3705321)
[![BibTeX](https://img.shields.io/badge/Cite-BibTeX-success.svg)](#citation)
[![LENA-1B](https://img.shields.io/badge/🤗%20Hugging%20Face-LENA--1B-yellow.svg)](https://huggingface.co/mhosseina96/LENA-1B)
[![LENA-3B](https://img.shields.io/badge/🤗%20Hugging%20Face-LENA--3B-yellow.svg)](https://huggingface.co/mhosseina96/LENA-3B)

> **LENA: Llama-based Embeddings of Neutralized Assembly for Cross-compiler/optimization Binary Code Similarity Detection**  
> Mohammadhossein Amouei, Benjamin C. M. Fung, Philippe Charland, and Jun Meng  
> *IEEE Transactions on Software Engineering*, pp. 1–24, 2026  
> **[Read the paper](https://doi.org/10.1109/TSE.2026.3705321)** · **[Download LENA-1B](https://huggingface.co/mhosseina96/LENA-1B)** · **[Download LENA-3B](https://huggingface.co/mhosseina96/LENA-3B)** · **[BibTeX](#citation)**

## Project Description

This is the official repository for **LENA**, a toolkit for cross-compiler and cross-optimization binary code similarity detection. LENA learns assembly-function representations using self-supervised learning and Llama-based language models.

## Getting Started

Follow the steps below to install LENA, prepare your dataset, download the pretrained models, and run the provided scripts.

### 1. Clone the Repository

```bash
git clone https://github.com/McGill-DMaS/LENA.git
cd LENA
```

### 2. Install Poetry

Install `pipx`, which provides an isolated environment for Python command-line applications:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

After installation, you may need to restart your shell or terminal before the `pipx` command becomes available.

Install Poetry through `pipx`:

```bash
pipx install poetry
```

Alternatively, consult the official [Poetry installation documentation](https://python-poetry.org/docs/#installation).

### 3. Install Project Dependencies

Install all project dependencies:

```bash
poetry install
```

Run commands inside the Poetry environment using either:

```bash
poetry run <command>
```

or activate the environment using the shell command supported by your Poetry installation.

Optionally, install the package in editable mode:

```bash
poetry run pip install -e .
```

## Data Preparation

Place the dataset under the `data/` directory using the following structure:

```text
data/
├── train/
├── validation/
└── test/
```

Each subdirectory should contain the binary functions, or their corresponding representations, to be processed by LENA.

## Pretrained Models

The pretrained LENA models are available on Hugging Face:

| Model | Repository |
|---|---|
| **LENA-1B** | [mhosseina96/LENA-1B](https://huggingface.co/mhosseina96/LENA-1B) |
| **LENA-3B** | [mhosseina96/LENA-3B](https://huggingface.co/mhosseina96/LENA-3B) |

Download the desired model and place its files under the `checkpoints/` directory.

A typical directory layout is:

```text
checkpoints/
├── LENA-1B/
└── LENA-3B/
```

## Build the Database

Generate the embedding database from the training dataset:

```bash
./build_database.sh
```

The script creates or updates the local database used for binary-function similarity search.

## Inference

Compute embeddings for the test dataset and perform similarity retrieval:

```bash
./inference.sh
```

By default, the inference results are saved to:

```text
data/inference_dict.pt
```

## Train the Pooler

Training the pooler is optional. To train it from scratch on your dataset, run:

```bash
./train_pooler.sh
```

The resulting checkpoints are saved under:

```text
checkpoints/pooler/
```

## Citation

If you use LENA, its pretrained models, or this repository in your research, please cite the following paper:

> M. Amouei, B. C. M. Fung, P. Charland, and J. Meng, “LENA: Llama-based Embeddings of Neutralized Assembly for Cross-compiler/optimization Binary Code Similarity Detection,” *IEEE Transactions on Software Engineering*, pp. 1–24, 2026, doi: 10.1109/TSE.2026.3705321.

<details>
<summary><strong>Show BibTeX</strong></summary>

```bibtex
@article{amouei2026lena,
  author   = {Amouei, Mohammadhossein and
              Fung, Benjamin C. M. and
              Charland, Philippe and
              Meng, Jun},
  title    = {{LENA: Llama-based Embeddings of Neutralized Assembly for
               Cross-compiler/optimization Binary Code Similarity Detection}},
  journal  = {IEEE Transactions on Software Engineering},
  year     = {2026},
  pages    = {1--24},
  doi      = {10.1109/TSE.2026.3705321},
  keywords = {Binary code similarity detection, self-supervised learning,
              assembly code embeddings, compiler-induced variations,
              tied similarity scores, vulnerability detection}
}
```

</details>

## Disclaimer

This software is provided as-is, without warranty or support. The authors assume no responsibility for damages, loss of income, or other problems arising from its use.

For further information, please consult the paper and the source code. Questions and issue reports may be submitted through the repository's issue tracker.
