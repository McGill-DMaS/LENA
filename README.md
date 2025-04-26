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

Download our pretrained model archive from Google Drive and extract it into the `checkpoints/` directory:

https://drive.google.com/file/d/1bCAv0kkl8CyjklG5nS4A0w-MYvU0fQm9/view?usp=sharing

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
./finetune.sh
```

This will run a training loop and save checkpoints under `checkpoints/pooler/`.

