# Subtask 2: Polarization Type Classification

## Overview

Multi-label classification task to identify the **types** of polarization in a text.

- **Input**: Text (English/Arabic)
- **Output**: 5 binary labels:
  - Political (0/1)
  - Racial/Ethnic (0/1)
  - Religious (0/1)
  - Gender/Sexual (0/1)
  - Other (0/1)
- **Models**: SVM (baseline), XLM-RoBERTa, RemBERT

## Important Notes

⚠️ **Multi-label**: A single text can have multiple polarization types simultaneously

Example:
```
Text: "Those immigrants are lazy Muslims"
Political: 0
Racial/Ethnic: 1 (mention of immigrants)
Religious: 1 (mention of Muslims)
Gender/Sexual: 0
Other: 0
```

## Data Format

```csv
id,text,political,racial/ethnic,religious,gender/sexual,other
eng_xxx,sample text,0,0,0,0,0
eng_yyy,polarized text,1,0,1,0,0
arb_xxx,نص عربي,0,1,0,0,1
```

## Quick Start

```bash
# Train all models
python src/train.py --config config/config.yaml --data-root data

# The script will:
# 1. Load train/ and dev/ data
# 2. Train SVM, XLM-RoBERTa, RemBERT
# 3. Evaluate with multi-label metrics
# 4. Save best models
```

## Configuration Parameters

### Data
- `train_path`: Path to training data
- `dev_path`: Path to development/validation data
- `languages`: Languages to load
- `multilabel: true`: Indicates multi-label task

### SVM Model (Multi-label)
- `multilabel_strategy: "one_vs_rest"`: Train one binary classifier per label
- Other parameters same as Subtask 1
- No probability estimates (all 0/1 predictions)

### Transformer Models (Multi-label)
- `model.problem_type: "multi_label_classification"`
- Uses binary cross-entropy with logits loss
- Sigmoid activation for probability outputs
- Threshold: 0.5 (configurable via `evaluation.threshold`)

### Label Configuration
```yaml
task:
  label_columns: 
    - "political"
    - "racial/ethnic"
    - "religious"
    - "gender/sexual"
    - "other"
```

## Output

### Models
- `models/svm/svm_multilabel_model.joblib`: SVM model + vectorizer
- `models/xlm_roberta/best_model/`: Transformer checkpoint
- `models/rembert/best_model/`: Transformer checkpoint

### Metrics Computed
- **Hamming Loss**: Fraction of labels that are incorrect
- **Subset Accuracy**: % of samples with all labels correct
- **Precision/Recall/F1**: Weighted across all labels
- **Per-label metrics**: Individual F1, precision, recall for each type
- **Per-language metrics**: Separate evaluation for English/Arabic

## Expected Performance

Baseline metrics from POLAR paper (F1-Score):

| Model | Political | Racial/Ethnic | Religious | Gender/Sexual | Other |
|-------|-----------|---------------|-----------|---------------|-------|
| SVM | 0.50 | 0.45 | 0.48 | 0.42 | 0.40 |
| XLM-RoBERTa | 0.58 | 0.52 | 0.55 | 0.50 | 0.48 |
| RemBERT | 0.62 | 0.58 | 0.60 | 0.55 | 0.52 |

## Configuration Examples

### Adjust Decision Threshold

```yaml
evaluation:
  threshold: 0.6  # Increase from 0.5 for more conservative predictions
```

Higher threshold → fewer positive predictions (higher precision, lower recall)

### Training Configuration

```yaml
xlm_roberta:
  training:
    batch_size: 16
    epochs: 5              # Multi-label often needs more epochs
    learning_rate: 2e-5
    loss_function: "bce_with_logits"  # Binary cross-entropy
    early_stopping_patience: 3
```

### Multi-label Specific

```yaml
xlm_roberta:
  model:
    problem_type: "multi_label_classification"
```

## Inference

Make predictions on new data:

```bash
python ../shared_utils/inference.py \
    --config config/config.yaml \
    --input-file test_texts.csv \
    --model-type xlm_roberta
```

Output format:
```csv
id,text,political,racial/ethnic,religious,gender/sexual,other,political_prob,racial/ethnic_prob,...
test_1,text,0,1,0,0,0,0.15,0.78,...
```

## Key Differences from Subtask 1

| Aspect | Subtask 1 (Binary) | Subtask 2 (Multi-label) |
|--------|-------------------|------------------------|
| Output | Single label | 5 binary labels per sample |
| Loss Function | Cross-entropy | Binary Cross-entropy |
| Activation | Softmax | Sigmoid |
| Metrics | Accuracy, Precision, Recall, F1 | Hamming Loss, Subset Accuracy, Per-label F1 |
| Threshold | Implicit (argmax) | Explicit (0.5 default) |
| Imbalance Handling | class_weight | More complex for multi-label |

## Tips for Best Results

1. **Threshold Tuning**: Test thresholds 0.3-0.7 via grid search
2. **Class Imbalance**: Some labels may be rarer, use higher epochs
3. **Per-label Performance**: Monitor individual label F1 scores
4. **Label Correlation**: Some labels co-occur more often (e.g., political + racial)
5. **Data Quality**: Multi-label annotations are harder, ensure quality

## Troubleshooting

**Q: High hamming loss despite good overall F1**
A: Some samples have incorrect label counts. Try:
- Lower learning_rate
- Increase epochs
- Adjust threshold

**Q: One label always predicts 0**
A: That label might be underrepresented. Check label distribution and:
- Increase training epochs
- Lower learning_rate
- Try class_weight balancing

**Q: Predictions are all 0s**
A: Models haven't converged. Try:
- Increase epochs
- Lower learning_rate (1e-5)
- Check data quality

## Advanced Configuration

### Custom Loss Weighting

Some labels might need more attention. You can adjust training weights in code (modify train.py):

```python
# In train.py, modify the loss computation
# This is research-level tuning
```

### Label-Specific Thresholds

Create different thresholds per label:

```python
# More conservative for rare labels
thresholds = {
    "political": 0.5,
    "racial/ethnic": 0.4,    # Lower threshold for rare label
    "religious": 0.5,
    "gender/sexual": 0.4,
    "other": 0.45
}
```

## References

- Paper: https://arxiv.org/pdf/2505.20624
- Multi-label Learning: https://scikit-learn.org/stable/modules/multiclass.html
- Transformers Multi-label: https://huggingface.co/docs/transformers/tasks/multi_label_classification

---

**Prerequisite**: Complete Subtask 1 first

**Next Step**: Train Subtask 3 (Manifestation Identification)
