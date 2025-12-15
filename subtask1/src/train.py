"""
Subtask 1: Polarization Detection (Binary Classification)
Training script for SVM, XLM-RoBERTa, and RemBERT models.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared_utils.utils import (
    ConfigManager, setup_logging, set_seed, 
    create_experiment_dir, print_config, get_device_info
)
from shared_utils.data_loader import DataLoader, TextPreprocessor


def train_svm_model(config: ConfigManager, train_data: pd.DataFrame, language: str, logger: logging.Logger) -> Dict:
    """Train SVM baseline model."""
    
    logger.info("="*50)
    logger.info(f"Training SVM Model for {language.upper()}")
    logger.info("="*50)
    
    from sklearn.svm import SVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    
   #Get configuration
    vectorizer_cfg = config.get("svm.vectorizer")
    svm_cfg = config.get("svm.svm_params")
    
   #Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=vectorizer_cfg['max_features'],
            ngram_range=tuple(vectorizer_cfg['ngram_range']),
            min_df=vectorizer_cfg['min_df'],
            max_df=vectorizer_cfg['max_df'],
            lowercase=vectorizer_cfg['lowercase'],
            stop_words=vectorizer_cfg['stop_words']
        )),
        ('svm', SVC(
            kernel=svm_cfg['kernel'],
            C=svm_cfg['C'],
            gamma=svm_cfg['gamma'],
            class_weight=svm_cfg['class_weight'],
            max_iter=svm_cfg['max_iter'],
            probability=svm_cfg['probability'],
            random_state=svm_cfg['random_state'],
            verbose=1
        ))
    ])
    
   #Train on full data (no validation split)
    logger.info(f"Training on {len(train_data)} samples...")
    pipeline.fit(train_data['text'], train_data['polarization'])
    
   #Save model with language suffix
    output_dir = config.get("svm.output_dir")
    os.makedirs(output_dir, exist_ok=True)
    
    import joblib
    model_path = os.path.join(output_dir, f"svm_{language}.joblib")
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return {
        "model": pipeline,
        "model_type": "svm"
    }


def train_transformer_model(
    config: ConfigManager,
    train_data: pd.DataFrame,
    model_name: str,
    language: str,
    logger: logging.Logger
) -> Dict:
    """Train XLM-RoBERTa or RemBERT model."""
    
    logger.info("="*50)
    logger.info(f"Training {model_name} Model for {language.upper()}")
    logger.info("="*50)
    
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    
   #Get model configuration
    if model_name == "xlm_roberta":
        model_cfg = config.get("xlm_roberta")
    elif model_name == "rembert":
        model_cfg = config.get("rembert")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
   #Load tokenizer and model
    pretrained_model = model_cfg['pretrained_model']
    logger.info(f"Loading pretrained model: {pretrained_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=config.get("task.num_classes")
    )
    model.to(device)
    
   #Create dataset
    class PolarizationDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts.iloc[idx])
            label = int(self.labels.iloc[idx])
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
   #Create data loaders (training data only, no validation)
    train_dataset = PolarizationDataset(
        train_data['text'],
        train_data['polarization'],
        tokenizer,
        model_cfg['tokenizer']['max_length']
    )
    
    batch_size = model_cfg['training']['batch_size']
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
   #Setup optimizer and scheduler
    num_epochs = model_cfg['training']['epochs']
    num_training_steps = len(train_loader) * num_epochs
    warmup_steps = model_cfg['training']['warmup_steps']
    
    learning_rate = float(model_cfg['training']['learning_rate'])
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
   #Training loop
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
       #Training
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg['training']['max_grad_norm'])
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
    
   #Save model with language suffix
    output_dir = model_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    language_output_dir = os.path.join(output_dir, language)
    model.save_pretrained(language_output_dir)
    tokenizer.save_pretrained(language_output_dir)
    logger.info(f"Model saved to {language_output_dir}")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_type": model_name
    }


def main(args):
    """Main training function."""
    
    set_seed(args.seed)
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=args.log_level
    )
    
    logger.info("Starting Subtask 1: Polarization Detection")
    logger.info(f"Device info: {get_device_info()}")
    
   #Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = ConfigManager(args.config)
    print_config(config, logger)
    
   #Load data - get parent directory from config path
    train_path = config.get("data.train_path")
    data_root = str(Path(train_path).parent) #Extract parent directory
    
    logger.info("Loading data...")
    data_loader = DataLoader(
        data_root=data_root,
        text_column=config.get("task.text_column"),
        label_columns=[config.get("task.label_column")],
        languages=config.get("data.languages"),
        random_seed=config.get("data.random_seed"),
        stratify=config.get("data.stratify")
    )
    
    train_data_dict, test_data_dict, _ = data_loader.load_all_data()
    test_data_combined = data_loader.combine_languages(test_data_dict, shuffle=False)
    
    logger.info(f"Test set size: {len(test_data_combined)}")
    
    results = {}
    
   #Helper function for making predictions
    def make_predictions_for_model(model_name, language, result, test_data_combined, config, logger):
        """Make predictions with a trained model on test data for specific language."""
        device = get_device_info()['device']
        predictions_dir = config.get(f"{model_name}.output_dir")
        os.makedirs(predictions_dir, exist_ok=True)
        
       #Only predict on the same language the model was trained on
        test_data_lang = test_data_combined[test_data_combined['language'] == language]
        if len(test_data_lang) == 0:
            logger.warning(f"No test data found for language: {language}")
            return
            
        if model_name == "svm":
           #SVM predictions - pipeline handles vectorization
            model = result["model"]
            y_pred = model.predict(test_data_lang['text'])
            y_prob = model.predict_proba(test_data_lang['text'])[:, 1]
        else:
           #Transformer predictions
            from torch.utils.data import DataLoader as TorchDataLoader, Dataset
            import torch
            
            class TextDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length=512):
                    self.texts = texts
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    encoding = self.tokenizer(
                        self.texts[idx],
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    return {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0)
                    }
            
            tokenizer = result['tokenizer']
            transformer_model = result['model']
            transformer_model.eval()
            
            y_pred_list = []
            y_prob_list = []
            
            dataset = TextDataset(test_data_lang['text'].tolist(), tokenizer)
            dataloader = TorchDataLoader(dataset, batch_size=32, shuffle=False)
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = transformer_model(input_ids, attention_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    
                    y_pred_list.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    y_prob_list.extend(probs[:, 1].cpu().numpy())
            
            y_pred = np.array(y_pred_list)
            y_prob = np.array(y_prob_list)
        
       #Save predictions for this language
        output_df = test_data_lang[['id', 'text']].copy()
        output_df['prediction'] = y_pred
        output_df['probability'] = y_prob
        
        pred_file = os.path.join(predictions_dir, f"test_predictions_{language}.csv")
        output_df.to_csv(pred_file, index=False)
        logger.info(f"{language.upper()} predictions ({model_name}) saved to {pred_file} ({len(output_df)} samples)")
    
   #Train models separately for each language and make predictions consecutively
    for language in config.get("data.languages"):
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING MODELS FOR LANGUAGE: {language.upper()}")
        logger.info(f"{'='*60}")
        
       #Get language-specific training data
        train_data_lang = train_data_dict.get(language, pd.DataFrame())
        if len(train_data_lang) == 0:
            logger.warning(f"No training data found for language: {language}")
            continue
        
        logger.info(f"Train set size ({language}): {len(train_data_lang)}")
        
       #Print statistics
        data_loader.print_statistics({"train": train_data_lang}, f"language_{language}")
        
       #Train and predict for each model type
        for model_name in ["svm", "xlm_roberta", "rembert"]:
            if model_name == "svm" and not config.get("svm.enabled"):
                continue
            elif model_name == "xlm_roberta" and not config.get("xlm_roberta.enabled"):
                continue
            elif model_name == "rembert" and not config.get("rembert.enabled"):
                continue
            
            logger.info(f"\n--- Training {model_name.upper()} for {language.upper()} ---")
            
            if model_name == "svm":
                result = train_svm_model(config, train_data_lang, language, logger)
            else:
                result = train_transformer_model(config, train_data_lang, model_name, language, logger)
            
            logger.info(f"\n--- Making predictions with {model_name.upper()} for {language.upper()} ---")
            make_predictions_for_model(model_name, language, result, test_data_combined, config, logger)
            
           #Store result
            key = f"{model_name}_{language}"
            results[key] = result
    
   #Summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        for metric, value in result['metrics'].items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for Subtask 1: Polarization Detection")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--data-root", type=str, default="data", help="Root data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    main(args)
