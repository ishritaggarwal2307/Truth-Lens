"""
Model Evaluation Script
========================
Comprehensive evaluation and visualization of model performance.
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

sys.path.append(str(Path(__file__).parent.parent))

from tensorflow import keras
from utils.config import get_config
from utils.audio_processing import AudioPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(config, preprocessor):
    """Load test dataset."""
    test_data_path = config.get('dataset.paths.raw_data', 'data/raw')
    
    X_mfcc = []
    y = []
    
    # Load real samples
    real_path = os.path.join(test_data_path, 'real')
    if os.path.exists(real_path):
        files = [f for f in os.listdir(real_path) if f.endswith(('.wav', '.mp3'))]
        logger.info(f"Loading {len(files)} real samples for testing...")
        
        for file in files[:50]:  # Limit to 50 for quick eval
            try:
                file_path = os.path.join(real_path, file)
                audio = preprocessor.load_audio(file_path)
                features = preprocessor.extract_features(audio)
                X_mfcc.append(features['mfcc'])
                y.append(0)
            except Exception as e:
                logger.warning(f"Error loading {file}: {e}")
    
    # Load fake samples
    fake_path = os.path.join(test_data_path, 'fake')
    if os.path.exists(fake_path):
        files = [f for f in os.listdir(fake_path) if f.endswith(('.wav', '.mp3'))]
        logger.info(f"Loading {len(files)} fake samples for testing...")
        
        for file in files[:50]:  # Limit to 50 for quick eval
            try:
                file_path = os.path.join(fake_path, file)
                audio = preprocessor.load_audio(file_path)
                features = preprocessor.extract_features(audio)
                X_mfcc.append(features['mfcc'])
                y.append(1)
            except Exception as e:
                logger.warning(f"Error loading {file}: {e}")
    
    if len(y) == 0:
        logger.error("No test data found!")
        return None, None
    
    X_mfcc = np.array(X_mfcc)
    y = np.array(y)
    
    # Add channel dimension
    X_mfcc = X_mfcc[..., np.newaxis]
    
    logger.info(f"Loaded {len(y)} test samples")
    return X_mfcc, y


def evaluate_model(model, X_test, y_test, output_dir='data/models/evaluation'):
    """Comprehensive model evaluation."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                   target_names=['Real', 'Fake'],
                                   output_dict=True)
    
    logger.info("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Save report
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    logger.info(f"ROC curve saved to {output_dir}/roc_curve.png")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    logger.info(f"PR curve saved to {output_dir}/precision_recall_curve.png")
    
    # Confidence distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_pred_proba[y_test == 0, 1], bins=30, alpha=0.7, label='Real samples', color='green')
    plt.hist(y_pred_proba[y_test == 1, 1], bins=30, alpha=0.7, label='Fake samples', color='red')
    plt.xlabel('Predicted Probability (Fake)')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    real_confidence = 1 - y_pred_proba[y_test == 0, 1]
    fake_confidence = y_pred_proba[y_test == 1, 1]
    
    plt.boxplot([real_confidence, fake_confidence], labels=['Real', 'Fake'])
    plt.ylabel('Confidence Score')
    plt.title('Confidence by True Class')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300)
    logger.info(f"Confidence distribution saved to {output_dir}/confidence_distribution.png")
    
    # Summary metrics
    metrics_summary = {
        'accuracy': float(report['accuracy']),
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1_score': float(report['weighted avg']['f1-score']),
        'roc_auc': float(roc_auc),
        'avg_precision': float(avg_precision),
        'total_samples': int(len(y_test)),
        'real_samples': int(np.sum(y_test == 0)),
        'fake_samples': int(np.sum(y_test == 1))
    }
    
    logger.info("\n=== Summary Metrics ===")
    for key, value in metrics_summary.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    
    # Save summary
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    return metrics_summary


def main():
    """Main evaluation function."""
    config = get_config()
    
    # Load model
    model_path = os.path.join(
        config.get('paths.models_dir', 'data/models'),
        'truth_lens_model.h5'
    )
    
    if not os.path.exists(model_path):
        model_path = os.path.join(
            config.get('paths.models_dir', 'data/models'),
            'best_model.h5'
        )
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first using: python src/train.py")
        return
    
    logger.info(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    
    # Load preprocessor
    preprocessor = AudioPreprocessor(config)
    
    # Load test data
    X_test, y_test = load_test_data(config, preprocessor)
    
    if X_test is None:
        logger.error("Could not load test data. Please add audio files to data/raw/")
        return
    
    # Evaluate
    logger.info("\nStarting comprehensive evaluation...")
    metrics = evaluate_model(model, X_test, y_test)
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to: data/models/evaluation/")
    logger.info("="*60)


if __name__ == "__main__":
    main()
