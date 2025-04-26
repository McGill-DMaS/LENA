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
curl -sSL https://install.python-poetry.org | python3 -
```

Ensure that `poetry` is in your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
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
bash build_database.sh
```

This will create (or update) the local database used for similarity search.

## ✨ Inference

To compute embeddings on the test set and perform retrieval:

```bash
bash inference.sh
```

Results will be saved in the `results/` directory by default.

## 🛠️ Train the Pooler (Optional)

If you wish to train the pooler network from scratch on your data:

```bash
bash finetune.sh
```

This will run a training loop and save checkpoints under `checkpoints/pooler/`.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

