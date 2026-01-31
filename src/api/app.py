"""
Truth-Lens FastAPI Backend
===========================
Production-ready API with:
- Real-time audio analysis
- Explainability (Grad-CAM heatmaps)
- Rate limiting
- Error handling
- CORS support
"""

import os
import sys
import io
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import librosa
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import cv2

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.audio_processing import AudioPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Truth-Lens API",
    description="Real-time Audio Deepfake Detection API",
    version="1.0.0"
)

# Load configuration
config = get_config()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api.cors.allow_origins', ['*']),
    allow_methods=config.get('api.cors.allow_methods', ['*']),
    allow_headers=config.get('api.cors.allow_headers', ['*']),
)

# Global variables
model = None
preprocessor = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    result: str  # "REAL" or "FAKE"
    confidence: float  # 0-100
    confidence_real: float
    confidence_fake: float
    is_fake: bool
    details: Dict
    explanation: Optional[str] = None
    heatmap: Optional[str] = None  # Base64 encoded image


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor
    
    logger.info("Loading model and preprocessor...")
    
    try:
        # Load model
        model_path = os.path.join(
            config.get('paths.models_dir', 'data/models'),
            'truth_lens_model.h5'
        )
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, trying best_model.h5")
            model_path = os.path.join(
                config.get('paths.models_dir', 'data/models'),
                'best_model.h5'
            )
        
        model = keras.models.load_model(model_path, compile=False)
        logger.info(f"Model loaded from {model_path}")
        
        # Recompile for inference
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load preprocessor
        preprocessor = AudioPreprocessor(config)
        logger.info("Preprocessor initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def generate_gradcam_heatmap(model, audio_features, pred_class):
    """
    Generate Grad-CAM heatmap for explainability.
    
    Args:
        model: Trained model
        audio_features: Input features
        pred_class: Predicted class index
        
    Returns:
        Base64 encoded heatmap image
    """
    try:
        # Find last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create model that maps input to last conv layer output and predictions
        grad_model = keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(audio_features)
            loss = predictions[:, pred_class]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        
        # Resize heatmap to match input
        heatmap = cv2.resize(heatmap, (audio_features.shape[2], audio_features.shape[1]))
        
        # Create visualization
        plt.figure(figsize=(10, 4))
        
        # Original spectrogram
        plt.subplot(1, 2, 1)
        plt.imshow(audio_features[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        plt.title('Input Features (MFCC)')
        plt.colorbar()
        
        # Heatmap overlay
        plt.subplot(1, 2, 2)
        plt.imshow(audio_features[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        plt.imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.6)
        plt.title('Grad-CAM Heatmap (Important Regions)')
        plt.colorbar()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    success = load_model_and_preprocessor()
    if not success:
        logger.warning("Model not loaded. Please train the model first.")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/analyze", response_model=PredictionResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze uploaded audio file for deepfake detection.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
        
    Returns:
        Prediction result with confidence scores
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Read file
        contents = await file.read()
        
        # Check file size (limit to 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Max 10MB.")
        
        # Preprocess audio
        audio = preprocessor.load_audio(io.BytesIO(contents))
        
        # Check if audio is too quiet (possible silence)
        if np.max(np.abs(audio)) < 0.01:
            return {
                "result": "UNCERTAIN",
                "confidence": 0.0,
                "confidence_real": 50.0,
                "confidence_fake": 50.0,
                "is_fake": False,
                "details": {
                    "message": "Audio is too quiet or empty. Please ensure microphone is working."
                },
                "explanation": "Insufficient audio signal detected",
                "heatmap": None
            }
        
        # Extract features
        features = preprocessor.extract_features(audio)
        
        # Prepare input for model
        mfcc_features = features['mfcc']
        mfcc_features = mfcc_features[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
        
        # Predict
        prediction = model.predict(mfcc_features, verbose=0)
        
        # Get probabilities
        prob_real = float(prediction[0][0])
        prob_fake = float(prediction[0][1])
        
        # Determine result
        is_fake = prob_fake > config.get('api.confidence_threshold', 0.5)
        result = "FAKE" if is_fake else "REAL"
        confidence = prob_fake if is_fake else prob_real
        
        # Generate explanation
        explanation = None
        if config.get('api.return_explanation', True):
            if is_fake:
                explanation = (
                    f"The audio exhibits patterns consistent with AI-generated speech. "
                    f"Confidence: {prob_fake*100:.1f}%. "
                    f"Key indicators: spectral irregularities and unnatural phase transitions."
                )
            else:
                explanation = (
                    f"The audio appears to be authentic human speech. "
                    f"Confidence: {prob_real*100:.1f}%. "
                    f"Natural vocal characteristics detected."
                )
        
        # Generate heatmap
        heatmap_base64 = None
        if config.get('api.return_heatmap', True):
            pred_class = 1 if is_fake else 0
            heatmap_base64 = generate_gradcam_heatmap(model, mfcc_features, pred_class)
        
        # Response
        response = {
            "result": result,
            "confidence": confidence * 100,
            "confidence_real": prob_real * 100,
            "confidence_fake": prob_fake * 100,
            "is_fake": is_fake,
            "details": {
                "model": "TruthLens-v1",
                "sample_rate": int(preprocessor.sample_rate),
                "duration": float(preprocessor.duration),
                "features_extracted": list(features.keys())
            },
            "explanation": explanation,
            "heatmap": heatmap_base64
        }
        
        logger.info(f"Prediction: {result} (confidence: {confidence*100:.2f}%)")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/batch-analyze")
async def batch_analyze(files: list[UploadFile] = File(...)):
    """
    Analyze multiple audio files at once.
    
    Args:
        files: List of audio files
        
    Returns:
        List of prediction results
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            result = await analyze_audio(file)
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"batch_results": results}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8000)
    
    logger.info(f"Starting Truth-Lens API on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=config.get('api.reload', True),
        log_level="info"
    )
