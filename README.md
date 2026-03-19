# LENA: Learning Embeddings of Native Assembly Using Llama with Self-supervision

This is the official repository for LENA, which is a toolkit for cross-compiler/optimization binary code similarity detection.

## 🚀 Getting Started

Follow these steps to set up the project, prepare your data, and run the provided scripts.

### 1. Clone the Repository

```bash
git clone https://github.com/McGill-DMaS/LENA.git
cd LENA
```

### 2. Install Poetry

If you don't have Poetry installed, follow the instructions at [python-poetry.org](https://python-poetry.org/) or run:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

Once installed, you may need to restart your shell or terminal for the `pipx` command to be available.

Then install Poetry:
```bash
pipx install poetry
```
### 3. Install Project Dependencies

Use Poetry to install all dependencies and enter the virtual environment:

```bash
poetry install
poetry shell
```

Optionally, make the package editable:

```bash
poetry run pip install -e .
```

## 📂 Data Preparation

Place your dataset under the `data/` directory with the following structure:

```
data/
├── train/
├── validation/
└── test/
```

Each subdirectory should contain the binary functions (or their representations) you wish to process.

## 📥 Download Pretrained Models

Download our pretrained models from Hugging Face repo and save it in the `checkpoints/` directory:

[LENA-1B](https://huggingface.co/mhosseina96/LENA-1B/tree/main)

[LENA-3B](https://huggingface.co/mhosseina96/LENA-3B/tree/main)



## 🔨 Build the Database

Run the script to generate the embeddings database from your training set:

```bash
./build_database.sh
```

This will create (or update) the local database used for similarity search.

## ✨ Inference

To compute embeddings on the test set and perform retrieval:

```bash
./inference.sh
```

Results will be saved in `data/inference_dict.pt` by default.

## 🛠️ Train the Pooler (Optional)

If you wish to train the pooler network from scratch on your data:

```bash
./train_pooler.sh
```

This will run a training loop and save checkpoints under `checkpoints/pooler/`.

## Disclaimer

The software is provided as-is with no warranty or support. We do not take any responsibility for any damage, loss of income, or any problems you might experience from using our software. If you have questions, you are encouraged to consult the paper and the source code. If you find our software useful, please cite our paper above.



