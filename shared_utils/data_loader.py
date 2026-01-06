"""
Data loading and preprocessing utilities for multilingual polarization detection.
Handles train/test/dev splits and data validation.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split
import logging


logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of multilingual data."""
    
    def __init__(
        self,
        data_root: str,
        text_column: str = "text",
        label_columns: Optional[List[str]] = None,
        id_column: str = "id",
        languages: Optional[List[str]] = None,
        random_seed: int = 42,
        stratify: bool = True
    ):
        """
        Initialize DataLoader.
        
        Args:
            data_root: Root directory containing data subdirectories
            text_column: Column name for text data
            label_columns: List of label column names (for multilabel)
            id_column: Column name for IDs
            languages: List of language codes to load (e.g., ['eng', 'arb'])
            random_seed: Random seed for reproducibility
            stratify: Whether to use stratified sampling
        """
        self.data_root = Path(data_root)
        self.text_column = text_column
        self.label_columns = label_columns or []
        self.id_column = id_column
        self.languages = languages or ['eng', 'arb']
        self.random_seed = random_seed
        self.stratify = stratify
        
        self.train_data = {}
        self.test_data = {}
        self.dev_data = {}
        self.val_data = {}
    
    def load_split(
        self,
        split_name: str,
        languages: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load data for a specific split (train/test/dev)."""
        
        if languages is None:
            languages = self.languages
        
        split_dir = self.data_root / split_name
        
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return {}
        
        split_data = {}
        
        for lang in languages:
            filepath = split_dir / f"{lang}.csv"
            
            if not filepath.exists():
                logger.warning(f"Language file not found: {filepath}")
                continue
            
            #Load CSV
            df = pd.read_csv(filepath)
            
            #Validate columns
            required_cols = [self.text_column, self.id_column]
            required_cols.extend(self.label_columns)
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {filepath}: {missing_cols}")
            
            #Clean text
            df[self.text_column] = df[self.text_column].astype(str).str.strip()
            df = df[df[self.text_column].str.len() > 0]
            
            #Add language column
            df['language'] = lang
            
            split_data[lang] = df
            logger.info(f"Loaded {len(df)} samples from {filepath}")
        
        return split_data
    
    def load_all_data(self) -> Tuple[Dict, Dict, Dict]:
        """Load train, test, and dev data."""
        
        self.train_data = self.load_split("train")
        self.test_data = self.load_split("test")
        self.dev_data = self.load_split("dev")
        
        return self.train_data, self.test_data, self.dev_data
    
    def combine_languages(
        self,
        data_dict: Dict[str, pd.DataFrame],
        shuffle: bool = True
    ) -> pd.DataFrame:
        """Combine data from multiple languages into a single DataFrame."""
        
        dfs = list(data_dict.values())
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        
        if shuffle:
            combined = combined.sample(frac=1.0, random_state=self.random_seed)
            combined = combined.reset_index(drop=True)
        
        return combined
    
    def create_train_val_split(
        self,
        data: pd.DataFrame,
        val_size: float = 0.1,
        stratify_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create validation split from training data."""
        
        if stratify_column and self.stratify and stratify_column in data.columns:
            stratify = data[stratify_column]
        else:
            stratify = None
        
        train, val = train_test_split(
            data,
            test_size=val_size,
            random_state=self.random_seed,
            stratify=stratify
        )
        
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        
        return train, val
    
    def get_statistics(self, data: pd.DataFrame) -> Dict:
        """Get dataset statistics."""
        
        stats = {
            "total_samples": len(data),
            "languages": data['language'].value_counts().to_dict() if 'language' in data.columns else {},
            "avg_text_length": data[self.text_column].str.split().str.len().mean(),
            "min_text_length": data[self.text_column].str.split().str.len().min(),
            "max_text_length": data[self.text_column].str.split().str.len().max(),
        }
        
        #Add label statistics for each label column
        if self.label_columns:
            for col in self.label_columns:
                if col in data.columns:
                    stats[f"{col}_distribution"] = data[col].value_counts().to_dict()
        
        return stats
    
    def print_statistics(self, data: Dict[str, pd.DataFrame], split_name: str) -> None:
        """Print statistics for a split."""
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Statistics for {split_name} split")
        logger.info(f"{'='*50}")
        
        for lang, df in data.items():
            logger.info(f"\nLanguage: {lang}")
            stats = self.get_statistics(df)
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    @staticmethod
    def clean_text(text: str, lowercase: bool = True, remove_extra_spaces: bool = True) -> str:
        """Clean text."""
        
        if not isinstance(text, str):
            return ""
        
        text = text.strip()
        
        if remove_extra_spaces:
            text = ' '.join(text.split())
        
        if lowercase:
            text = text.lower()
        
        return text
    
    @staticmethod
    def get_text_statistics(text: str) -> Dict:
        """Get statistics for a text."""
        
        tokens = text.split()
        
        return {
            "length": len(text),
            "token_count": len(tokens),
            "avg_token_length": np.mean([len(t) for t in tokens]) if tokens else 0,
            "unique_tokens": len(set(tokens))
        }
