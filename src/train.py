"""
Truth-Lens Model Training Script
=================================
Complete training pipeline with:
- Data loading and preprocessing
- Model training with callbacks
- Evaluation and metrics
- Model saving and export
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
import pickle

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.audio_processing import AudioPreprocessor
from models.ensemble_model import build_ensemble_model, build_lightweight_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare dataset for training."""
    
    def __init__(self, config, preprocessor):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
            preprocessor: AudioPreprocessor instance
        """
        self.config = config
        self.preprocessor = preprocessor
        self.data_path = config.get('dataset.paths.raw_data', 'data/raw')
        
    def load_dataset(self):
        """
        Load dataset from directory structure:
        data/raw/
            real/
                file1.wav
                file2.wav
                ...
            fake/
                file1.wav
                file2.wav
                ...
        
        Returns:
            Tuple of (X, y) where X is dict of features and y is labels
        """
        logger.info("Loading dataset...")
        
        X_mfcc = []
        X_mel = []
        y = []
        
        # Load real audio
        real_path = os.path.join(self.data_path, 'real')
        if os.path.exists(real_path):
            real_files = [f for f in os.listdir(real_path) if f.endswith(('.wav', '.mp3'))]
            logger.info(f"Found {len(real_files)} real audio files")
            
            for idx, file in enumerate(real_files):
                try:
                    file_path = os.path.join(real_path, file)
                    audio = self.preprocessor.load_audio(file_path)
                    features = self.preprocessor.extract_features(audio)
                    
                    X_mfcc.append(features['mfcc'])
                    if 'mel_spectrogram' in features:
                        X_mel.append(features['mel_spectrogram'])
                    
                    y.append(0)  # 0 = Real
                    
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processed {idx + 1}/{len(real_files)} real files")
                        
                except Exception as e:
                    logger.warning(f"Error processing {file}: {e}")
        
        # Load fake audio
        fake_path = os.path.join(self.data_path, 'fake')
        if os.path.exists(fake_path):
            fake_files = [f for f in os.listdir(fake_path) if f.endswith(('.wav', '.mp3'))]
            logger.info(f"Found {len(fake_files)} fake audio files")
            
            for idx, file in enumerate(fake_files):
                try:
                    file_path = os.path.join(fake_path, file)
                    audio = self.preprocessor.load_audio(file_path)
                    features = self.preprocessor.extract_features(audio)
                    
                    X_mfcc.append(features['mfcc'])
                    if 'mel_spectrogram' in features:
                        X_mel.append(features['mel_spectrogram'])
                    
                    y.append(1)  # 1 = Fake
                    
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processed {idx + 1}/{len(fake_files)} fake files")
                        
                except Exception as e:
                    logger.warning(f"Error processing {file}: {e}")
        
        # Convert to numpy arrays
        X_mfcc = np.array(X_mfcc)
        y = np.array(y)
        
        # Add channel dimension for CNN
        X_mfcc = X_mfcc[..., np.newaxis]
        
        logger.info(f"Dataset loaded: {len(y)} samples")
        logger.info(f"Real samples: {np.sum(y == 0)}, Fake samples: {np.sum(y == 1)}")
        logger.info(f"MFCC shape: {X_mfcc.shape}")
        
        # Prepare feature dict
        X_dict = {'mfcc': X_mfcc}
        
        if len(X_mel) > 0:
            X_mel = np.array(X_mel)
            X_mel = X_mel[..., np.newaxis]
            X_dict['mel'] = X_mel
            logger.info(f"Mel-Spectrogram shape: {X_mel.shape}")
        
        return X_dict, y


def prepare_data(X_dict, y, config):
    """
    Split data into train/val/test sets.
    
    Args:
        X_dict: Dictionary of feature arrays
        y: Labels
        config: Configuration object
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    train_split = config.get('dataset.train_split', 0.7)
    val_split = config.get('dataset.val_split', 0.15)
    test_split = config.get('dataset.test_split', 0.15)
    random_seed = config.get('dataset.random_seed', 42)
    
    # Convert labels to categorical
    y_cat = keras.utils.to_categorical(y, num_classes=2)
    
    # First split: train + val vs test
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        random_state=random_seed,
        stratify=y
    )
    
    # Second split: train vs val
    y_train_val = y[train_val_idx]
    val_size = val_split / (train_split + val_split)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_seed,
        stratify=y_train_val
    )
    
    # Split each feature array
    X_train = {key: val[train_idx] for key, val in X_dict.items()}
    X_val = {key: val[val_idx] for key, val in X_dict.items()}
    X_test = {key: val[test_idx] for key, val in X_dict.items()}
    
    y_train = y_cat[train_idx]
    y_val = y_cat[val_idx]
    y_test = y_cat[test_idx]
    
    logger.info(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_callbacks(config):
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(
        config.get('paths.models_dir', 'data/models'),
        'best_model.h5'
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    if config.get('training.early_stopping.enabled', True):
        early_stop = keras.callbacks.EarlyStopping(
            monitor=config.get('training.early_stopping.monitor', 'val_accuracy'),
            patience=config.get('training.early_stopping.patience', 10),
            restore_best_weights=config.get('training.early_stopping.restore_best_weights', True),
            verbose=1
        )
        callbacks.append(early_stop)
    
    # Learning rate scheduler
    if config.get('training.lr_schedule.enabled', True):
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.get('training.lr_schedule.factor', 0.5),
            patience=config.get('training.lr_schedule.patience', 5),
            min_lr=config.get('training.lr_schedule.min_lr', 0.00001),
            verbose=1
        )
        callbacks.append(lr_scheduler)
    
    # TensorBoard
    log_dir = os.path.join(
        config.get('paths.logs_dir', 'logs'),
        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    callbacks.append(tensorboard)
    
    logger.info(f"Callbacks configured. Logs: {log_dir}")
    
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, config):
    """
    Train the model.
    
    Args:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Configuration object
        
    Returns:
        Training history
    """
    batch_size = config.get('training.batch_size', 32)
    epochs = config.get('training.epochs', 50)
    
    # Convert dict to list for multi-input models
    if isinstance(X_train, dict):
        X_train_list = [X_train['mfcc']]
        X_val_list = [X_val['mfcc']]
        
        if 'mel' in X_train:
            X_train_list.append(X_train['mel'])
            X_val_list.append(X_val['mel'])
    else:
        X_train_list = X_train
        X_val_list = X_val
    
    callbacks = get_callbacks(config)
    
    logger.info("Starting training...")
    
    history = model.fit(
        X_train_list,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_list, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed!")
    
    return history


def evaluate_model(model, X_test, y_test, config):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        config: Configuration object
    """
    logger.info("Evaluating model on test set...")
    
    # Convert dict to list
    if isinstance(X_test, dict):
        X_test_list = [X_test['mfcc']]
        if 'mel' in X_test:
            X_test_list.append(X_test['mel'])
    else:
        X_test_list = X_test
    
    # Evaluate
    results = model.evaluate(X_test_list, y_test, verbose=1)
    
    logger.info("Test Results:")
    for name, value in zip(model.metrics_names, results):
        logger.info(f"  {name}: {value:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test_list)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    report = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=['Real', 'Fake']
    )
    logger.info("\nClassification Report:\n" + report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    output_dir = config.get('paths.models_dir', 'data/models')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    
    return results, report, cm


def save_training_info(history, config):
    """Save training history and configuration."""
    output_dir = config.get('paths.models_dir', 'data/models')
    
    # Save history
    history_path = os.path.join(output_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    # Save config
    config_path = os.path.join(output_dir, 'model_config.json')
    # Convert config to dict for JSON serialization
    config_dict = config.config
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Training info saved to {output_dir}")


def plot_training_history(history, config):
    """Plot training curves."""
    output_dir = config.get('paths.models_dir', 'data/models')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    logger.info(f"Training curves saved to {output_dir}/training_curves.png")


def main():
    """Main training pipeline."""
    # Load configuration
    config = get_config()
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(config)
    
    # Load dataset
    data_loader = DataLoader(config, preprocessor)
    X_dict, y = data_loader.load_dataset()
    
    if len(y) == 0:
        logger.error("No data found! Please add audio files to data/raw/real and data/raw/fake")
        return
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X_dict, y, config)
    
    # Build model (use lightweight for faster training)
    logger.info("Building model...")
    model = build_lightweight_model(config)
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, config)
    
    # Save training info
    save_training_info(history, config)
    plot_training_history(history, config)
    
    # Save final model
    final_model_path = os.path.join(
        config.get('paths.models_dir', 'data/models'),
        'truth_lens_model.h5'
    )
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
