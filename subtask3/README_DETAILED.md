# Subtask 3: Manifestation Identification

## Overview

Multi-label classification task to identify **how** polarization is expressed in a text.

- **Input**: Text (English/Arabic)
- **Output**: 6 binary labels indicating manifestation types:
  - Stereotype (0/1)
  - Vilification (0/1)
  - Dehumanization (0/1)
  - Extreme Language (0/1)
  - Lack of Empathy (0/1)
  - Invalidation (0/1)
- **Models**: SVM (baseline), XLM-RoBERTa, RemBERT

## Manifestation Types Explained

| Type | Definition | Example |
|------|-----------|---------|
| **Stereotype** | Generalizing characteristics to a whole group | "All X are lazy" |
| **Vilification** | Using derogatory language toward a group | "X are despicable" |
| **Dehumanization** | Portraying group as less than human | "X are animals" |
| **Extreme Language** | Intense, exaggerated statements | "X are the worst things ever" |
| **Lack of Empathy** | Not acknowledging group's perspective/feelings | Ignoring X's concerns |
| **Invalidation** | Denying group's experiences or validity | "X's problems don't matter" |

## Data Format

```csv
id,text,stereotype,vilification,dehumanization,extreme_language,lack_of_empathy,invalidation
eng_xxx,sample text,0,0,0,0,0,0
eng_yyy,polarized text,1,1,0,1,0,0
arb_xxx,نص عربي,0,1,1,1,1,1
```

## Quick Start

```bash
# Train all models
python src/train.py --config config/config.yaml --data-root data

# The script will:
# 1. Load train/ and dev/ data
# 2. Train SVM, XLM-RoBERTa, RemBERT for 6-label multi-label classification
# 3. Compute per-label metrics
# 4. Save trained models
```

## Configuration Parameters

### Data Configuration
```yaml
data:
  train_path: "data/train"
  dev_path: "data/dev"
  languages: ["eng", "arb"]
  multilabel: true
```

### Task Configuration
```yaml
task:
  task_type: "multi_label_classification"
  num_classes: 6
  label_columns:
    - "stereotype"
    - "vilification"
    - "dehumanization"
    - "extreme_language"
    - "lack_of_empathy"
    - "invalidation"
```

### Model-Specific Parameters

**SVM:**
```yaml
svm:
  multilabel_strategy: "one_vs_rest"
  vectorizer:
    max_features: 5000
    ngram_range: [1, 2]
```

**Transformers:**
```yaml
xlm_roberta:
  model:
    num_labels: 6
    problem_type: "multi_label_classification"
  training:
    batch_size: 16
    epochs: 3
    learning_rate: 2e-5
    loss_function: "bce_with_logits"
```

## Output & Metrics

### Trained Models
- `models/svm/svm_multilabel_model.joblib`
- `models/xlm_roberta/best_model/`
- `models/rembert/best_model/`

### Evaluation Metrics
- **Hamming Loss**: % of incorrect labels
- **Subset Accuracy**: % of perfectly correct samples
- **Precision/Recall/F1**: Averaged across labels
- **Per-label metrics**: Individual performance for each manifestation
- **Per-language metrics**: Separate English/Arabic evaluation

### Log Files
- `logs/training_*.log`: Complete training logs
- Console output: Real-time metrics during training

## Expected Performance

Baseline results from POLAR paper (Weighted F1):

| Model | Stereotype | Vilification | Dehumanization | Extreme_Lang | Lack_Empathy | Invalidation |
|-------|-----------|--------------|-----------------|-------------|--------------|-------------|
| SVM | 0.42 | 0.48 | 0.38 | 0.45 | 0.35 | 0.40 |
| XLM-RoBERTa | 0.55 | 0.60 | 0.52 | 0.58 | 0.48 | 0.53 |
| RemBERT | 0.62 | 0.68 | 0.60 | 0.65 | 0.55 | 0.60 |

## Key Features

- Multi-label Support: Handle multiple simultaneous manifestations
- Per-label Analysis: Understand which manifestations are detected well
- Language-specific: Evaluate English and Arabic separately
- Flexible Thresholds: Adjust confidence threshold per manifestation type
- Research Metrics: Comprehensive evaluation suite

## Configuration Examples

### Adjust Decision Threshold

```yaml
evaluation:
  threshold: 0.5  # Default
  # Higher → fewer positives (higher precision)
  # Lower → more positives (higher recall)
```

### Language-Specific Evaluation

```yaml
evaluation:
  report_per_language: true  # Enable
  save_predictions: true
```

Metrics will be computed for:
- Overall
- English
- Arabic

### Training Optimization

```yaml
xlm_roberta:
  training:
    batch_size: 32              # Larger batches
    gradient_accumulation_steps: 1
    epochs: 5                   # More epochs for complex task
    learning_rate: 1.5e-5       # Slightly lower LR
    warmup_steps: 1000
    early_stopping_patience: 3
```

## Label Correlation Analysis

Different manifestations often co-occur:

```
Stereotype ↔ Vilification (both negative group characterization)
Dehumanization ↔ Vilification (intensity of attacks)
Extreme Language ↔ Most others (tends to amplify)
Lack of Empathy ↔ Invalidation (dismissal patterns)
```

Consider this when interpreting per-label performance.

## Inference

Make predictions on new data:

```bash
python ../shared_utils/inference.py \
    --config config/config.yaml \
    --input-file test_data.csv \
    --model-type xlm_roberta
```

Input CSV:
```csv
id,text
test_1,sample text
test_2,another text
```

Output CSV:
```csv
id,text,stereotype,vilification,dehumanization,extreme_language,lack_of_empathy,invalidation,stereotype_prob,...
test_1,sample,0,1,0,1,0,0,0.12,0.78,...
```

## Differences from Previous Subtasks

| Aspect | Subtask 1 | Subtask 2 | Subtask 3 |
|--------|-----------|-----------|-----------|
| Task Type | Binary | Multi-label (5) | Multi-label (6) |
| Output Classes | 2 | 5 | 6 |
| Focus | Presence of polarization | Type of polarization | Expression method |
| Dependency | None | Subtask 1 (optional) | Subtask 2 (optional) |
| Complexity | Moderate | High | Highest |

## Tips for Best Results

1. **Label Imbalance**: Some manifestations may be rarer. Use:
   - Higher epochs (5-10)
   - Lower learning rate (1e-5)
   - Adjusted thresholds per label

2. **Manifestation Correlation**: Exploit label dependencies:
   - Don't treat labels as independent
   - Review samples with unusual label combinations

3. **Threshold Optimization**:
   ```python
   # Grid search optimal thresholds
   thresholds = np.arange(0.3, 0.7, 0.05)
   # Evaluate each threshold value
   ```

4. **Error Analysis**:
   - Look at false positives/negatives per manifestation
   - Some manifestations harder to detect than others
   - Check per-language performance differences

5. **Data Quality**:
   - 6-label annotation is complex
   - Check inter-annotator agreement
   - Verify label distributions are reasonable

## Advanced Configurations

### Custom Training for Rare Labels

```python
# In train.py, modify loss weighting (research-level):
label_weights = {
    "stereotype": 1.0,
    "vilification": 1.0,
    "dehumanization": 1.5,      # Upweight rare/hard labels
    "extreme_language": 1.0,
    "lack_of_empathy": 1.5,
    "invalidation": 1.2
}
```

### Per-Label Thresholds

```python
# Different threshold per manifestation
threshold_config = {
    "stereotype": 0.5,
    "vilification": 0.5,
    "dehumanization": 0.4,      # Lower for rare class
    "extreme_language": 0.5,
    "lack_of_empathy": 0.4,
    "invalidation": 0.45
}
```

## Troubleshooting

**Q: Some manifestation always predicts 0**
A: Check label prevalence in training data. If rare:
- Use lower threshold for that label
- Increase epochs
- Check data quality

**Q: High hamming loss but decent F1**
A: Some samples have incorrect label counts. Try:
- Increase epochs to 5-10
- Reduce learning_rate to 1e-5
- Check for mislabeled data

**Q: Validation loss not decreasing**
A: Model not converging. Try:
- Much lower learning_rate (5e-6)
- More warmup steps
- Check data quality

## Evaluation Script

For detailed analysis, create a simple evaluation script:

```python
from shared_utils.metrics import MultiLabelMetrics

metrics = MultiLabelMetrics()
y_true = ...  # Ground truth labels
y_pred = ...  # Model predictions

overall = metrics.compute(y_true, y_pred)
per_label = metrics.get_per_label_metrics(y_true, y_pred, label_names)

print(f"Overall F1: {overall['f1']:.4f}")
for label, scores in per_label.items():
    print(f"{label}: F1={scores['f1']:.4f}")
```

## References

- Paper: https://arxiv.org/pdf/2505.20624
- Multi-label Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
- Manifestation Analysis: https://arxiv.org/pdf/2505.20624 (See Table in paper)

## Dataset Statistics

Check your data:

```python
import pandas as pd

df = pd.read_csv("data/train/eng.csv")
print(df[["stereotype", "vilification", "dehumanization", 
          "extreme_language", "lack_of_empathy", "invalidation"]].sum())
# Shows count of each manifestation type
```

---

**Prerequisites**: Complete Subtasks 1 & 2 first (optional but recommended)

**Final Step**: Use predictions for Codabench evaluation
