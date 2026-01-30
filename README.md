# Low-Resource ASR

ASR for Endangered Languages using Mozilla Common Voice Spontaneous Speech datasets.

## Overview

This project evaluates and improves automatic speech recognition (ASR) systems for 21 underrepresented languages from Africa, Asia, Europe, and the Americas. It establishes baselines using MMS and Whisper, implements improved architectures, and conducts linguistic error analysis.

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/low-resource-asr.git
cd low-resource-asr

# Create virtual environment and install dependencies
uv sync

# For development dependencies
uv sync --all-extras
```

### Activate Environment

```bash
# Activate the virtual environment
source .venv/bin/activate
```

## Project Structure

```
low-resource-asr/
├── data/
│   ├── mozilla_speech_data/    # Mozilla Common Voice datasets
│   └── linguistic_resources/   # Grammars, phoneme inventories
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model definitions and configs
│   ├── training/               # Training scripts
│   └── evaluation/             # Evaluation and error analysis
├── notebooks/                  # Jupyter notebooks for exploration
├── scripts/                    # CLI scripts
├── results/                    # Experiment results
├── .notes/                     # Project documentation
├── pyproject.toml              # Dependencies (uv)
└── README.md
```

## Languages

21 languages across 4 regions:

| Region | Languages |
|--------|-----------|
| Africa | Bukusu, Chiga, Nubi, Konzo, Lendu, Kenyi, Thur, Ruuli, Amba, Rutoro, Kuku |
| Americas | Wixárika, Southwestern Tlaxiaco Mixtec, Michoacán Mazahua, Papantla Totonac, Toba Qom |
| Europe | Gheg Albanian, Cypriot Greek, Scots |
| Asia | Betawi, Western Penan |

## Usage

```bash
# Run baseline evaluation
uv run python scripts/run_baseline.py

# Fine-tune model
uv run python scripts/finetune.py --model whisper-small --lang aln

# Evaluate model
uv run python scripts/evaluate.py --model-path models/whisper-small-aln
```

## Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name
```

## License

MIT
