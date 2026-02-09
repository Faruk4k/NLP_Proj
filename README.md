# POLAR: Multilingual Polarization Detection 

##  Overview
three interconnected subtasks:

1. **Subtask 1**: Polarization Detection (Binary Classification)
   - Detect if text contains polarized content
   - Models: SVM, XLM-RoBERTa, RemBERT

2. **Subtask 2**: Polarization Type Classification (Multi-label)
   - Classify types: Political, Racial/Ethnic, Religious, Gender/Sexual, Other
   - Multi-label classification with 5 categories

3. **Subtask 3**: Manifestation Identification (Multi-label)
   - Identify how polarization is expressed
   - 6 manifestation types: Stereotype, Vilification, Dehumanization, Extreme Language, Lack of Empathy, Invalidation

## Models

We use three models for all subtasks:

1. **SVM (Statistical Model - Baseline)**
   - TF-IDF feature extraction
   - Fast training and inference
   - Good baseline for comparison

2. **XLM-RoBERTa** (122M parameters)
   - Multilingual transformer (100+ languages)
   - Good balance of quality and speed
   - ~30-60 min training on GPU

3. **RemBERT** (568M parameters)
   - Google's Retrieval-based multilingual BERT
   - Highest quality predictions
   - ~1-2 hours training on GPU




## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows version: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train First Model

```bash
# Run from project root (not from subtask1 directory)
python subtask1/src/train.py --config subtask1/config/config.yaml
```

### View Results

```bash
# Check logs
tail logs/training_*.log

# View trained models
ls models/
```

## Project Structure

```
NLP_Proj/
├── subtask1/               # Polarization Detection
│   ├── config/config.yaml  
│   ├── data/              
│   ├── src/train.py        # script
│   ├── models/             
│   └── logs/               
├── subtask2/               # Polarization Type (Multi-label)
├── subtask3/               # Manifestation Identification (Multi-label)
├── shared_utils/           # Shared utilities
│   ├── utils.py            # Configuration, logging
│   ├── data_loader.py      # Data loading & preprocessing
│   ├── metrics.py          # Evaluation metrics
│   └── inference.py        # Inference utilities
├── requirements.txt        # Python dependencies
└── README.md              
```

## Configuration System

**All parameters are configurable** via YAML files :

### SVM Configuration  --- No need to change this I think (?)
- Feature extraction (TF-IDF settings)
- Kernel selection (linear, rbf, poly, sigmoid)
- Regularization parameters

### Transformer Configuration (XLM-RoBERTa & RemBERT)
- Model selection
- Tokenizer settings (max_length, padding)
- Training parameters:
  - Batch size
  - Learning rate & scheduler
  - Number of epochs
  - Early stopping
  - Gradient accumulation
  - Optimizer selection
- Mixed precision training
- GPU/CPU device configuration


## Training

### Subtask 1 (Polarization Detection)

```bash
python subtask1/src/train.py \
    --config subtask1/config/config.yaml \
    --data-root subtask1/data \
    --seed 42
```

**Output:**
- Trained models in `subtask1/models/`
- Training logs in `subtask1/logs/`
- Per-language metrics

### Subtask 2 (Polarization Type)

```bash
python subtask2/src/train.py --config subtask2/config/config.yaml
```

Multi-label classification for 5 polarization types

### Subtask 3 (Manifestation Identification)

```bash
python subtask3/src/train.py --config subtask3/config/config.yaml
```

Multi-label classification for 6 manifestation types


## Make Predictions

```bash
python shared_utils/inference.py \
    --config subtask1/config/config.yaml \
    --input-file new_data.csv \
    --model-type xlm_roberta \
    --output-dir predictions/
```

## References

- Competition: https://www.codabench.org/competitions/10669/
- XLM-RoBERTa: https://huggingface.co/xlm-roberta-base
- RemBERT: https://huggingface.co/google/rembert