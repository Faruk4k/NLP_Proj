# Subtask 1: Polarization Detection

## Overview

Binary classification task to detect whether a text contains polarized content.

- **Input**: Text (English/Arabic)
- **Output**: Binary label (0 = non-polarized, 1 = polarized)
- **Models**: SVM (baseline), XLM-RoBERTa, RemBERT

## Data Format

```csv
id,text,polarization
eng_xxx,sample text here,0
eng_yyy,another polarized text,1
arb_xxx,نص عربي,0
```

## Quick Start

```bash
# Train all models
python src/train.py --config config/config.yaml --data-root data

# Train only specific models (edit config.yaml):
# svm.enabled: true
# xlm_roberta.enabled: true
# rembert.enabled: false
```

## Configuration Parameters

All parameters are in `config/config.yaml`:

### Data
- `train_path`: Path to training data
- `test_path`: Path to test data  
- `languages`: List of languages to load
- `random_seed`: Seed for reproducibility
- `stratify`: Use stratified sampling

### SVM Model
- `vectorizer.max_features`: Maximum TF-IDF features (default: 5000)
- `vectorizer.ngram_range`: N-gram range (default: 1-2)
- `svm_params.kernel`: SVM kernel (rbf, linear, poly, sigmoid)
- `svm_params.C`: Regularization strength
- `svm_params.gamma`: Kernel coefficient

### Transformer Models (XLM-RoBERTa, RemBERT)
- `training.batch_size`: Batch size (default: 16)
- `training.epochs`: Number of epochs (default: 3)
- `training.learning_rate`: Learning rate (default: 2e-5)
- `training.early_stopping`: Enable early stopping
- `training.early_stopping_patience`: Patience for early stopping
- `tokenizer.max_length`: Maximum sequence length

## Output

### Models
Trained models are saved in:
- `models/svm/svm_model.joblib`
- `models/xlm_roberta/best_model/`
- `models/rembert/best_model/`

### Logs
- `logs/training_*.log`: Training logs with metrics
- Console output: Real-time training progress

## Metrics

The evaluation computes:
- **Accuracy**: Overall correctness
- **Precision**: True positives / predicted positives
- **Recall**: True positives / actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: TP, FP, FN, TN breakdown

Per-language metrics are also reported separately.

## Expected Performance

Based on the POLAR paper benchmarks:

| Model | F1-Score (Avg) |
|-------|---|
| SVM | ~0.55 |
| XLM-RoBERTa | ~0.62 |
| RemBERT | ~0.65 |

Your results may vary based on data quality and configuration.

## Tips for Best Results

1. **Data Imbalance**: If classes are imbalanced, the SVM config uses `class_weight: "balanced"`
2. **Text Length**: Set `tokenizer.max_length` based on your data distribution
3. **Learning Rate**: Try 5e-5 or 1e-5 for transformers if overfitting
4. **Batch Size**: Increase if GPU memory allows for faster training
5. **Early Stopping**: Set `early_stopping: true` and tune `early_stopping_patience`

## Inference

Make predictions on new data:

```bash
python ../shared_utils/inference.py \
    --config config/config.yaml \
    --input-file test_data.csv \
    --model-type xlm_roberta
```

Input CSV format:
```csv
id,text
test_1,text to classify
test_2,another text
```

Output: `predictions/predictions.csv` with predictions and confidences

## Advanced Configuration

### Custom Hyperparameter Tuning

Edit `config/config.yaml`:

```yaml
# Try lower learning rate for better convergence
xlm_roberta:
  training:
    learning_rate: 1e-5
    warmup_steps: 1000
    weight_decay: 0.01
    
# Try larger batch size with gradient accumulation
xlm_roberta:
  training:
    batch_size: 8
    gradient_accumulation_steps: 2  # Effective: 16
```

### Use Different Pretrained Models

```yaml
xlm_roberta:
  pretrained_model: "xlm-roberta-large"  # Use large variant
  
rembert:
  pretrained_model: "google/rembert"     # Already uses best version
```

## Troubleshooting

**Q: Model crashes with "CUDA out of memory"**
A: Reduce `batch_size` to 8 or 4 in config.yaml

**Q: Training is very slow**
A: 
- Increase `batch_size` (if memory allows)
- Reduce `epochs` to 1 or 2 for testing
- Set `rembert.enabled: false`

**Q: Poor F1 score**
A:
- Check data quality and label distribution
- Increase `epochs` to 5-10
- Try different `learning_rate` values
- Increase `warmup_steps` to 1000 or more

## References

- Paper: https://arxiv.org/pdf/2505.20624
- XLM-RoBERTa: https://huggingface.co/xlm-roberta-base
- RemBERT: https://huggingface.co/google/rembert

---

**Next Step**: Train Subtask 2 (Polarization Type Classification)
