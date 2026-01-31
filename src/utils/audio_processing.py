"""
Advanced Audio Preprocessing for Truth-Lens
============================================
Multi-feature extraction: MFCC, Mel-Spectrogram, Raw Waveform, Spectral Features
"""

import numpy as np
import librosa
import logging
from typing import Dict, Optional, Tuple
from scipy.signal import butter, lfilter

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Advanced audio preprocessing with multiple feature extraction methods.
    
    Features:
    - MFCC (Mel-Frequency Cepstral Coefficients)
    - Mel-Spectrogram
    - Raw Waveform (normalized)
    - Spectral Features (centroid, rolloff, ZCR)
    """
    
    def __init__(self, config):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.sample_rate = config.get('audio.sample_rate', 16000)
        self.duration = config.get('audio.duration', 3.0)
        self.hop_length = config.get('audio.hop_length', 512)
        
        # Feature configs
        self.mfcc_config = config.get('model.features.mfcc', {})
        self.mel_config = config.get('model.features.mel_spectrogram', {})
        self.spectral_config = config.get('model.features.spectral', {})
        
        # Preprocessing options
        self.normalize = config.get('audio.preprocessing.normalize', True)
        self.remove_silence = config.get('audio.preprocessing.remove_silence', True)
        self.silence_threshold = config.get('audio.preprocessing.silence_threshold', 0.01)
        self.apply_preemphasis = config.get('audio.preprocessing.apply_preemphasis', True)
        self.preemphasis_coef = config.get('audio.preprocessing.preemphasis_coef', 0.97)
        
        logger.info("AudioPreprocessor initialized")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file or file-like object
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Load audio
            audio, sr = librosa.load(
                audio_path, 
                sr=self.sample_rate, 
                duration=self.duration
            )
            
            # Ensure fixed length
            audio = self._fix_length(audio)
            
            # Apply preprocessing steps
            if self.remove_silence:
                audio = self._remove_silence(audio)
                audio = self._fix_length(audio)  # Re-fix length after silence removal
            
            if self.apply_preemphasis:
                audio = self._apply_preemphasis(audio)
            
            if self.normalize:
                audio = self._normalize(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract multiple features from audio.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary of feature arrays
        """
        features = {}
        
        # MFCC Features
        if self.mfcc_config.get('enabled', True):
            features['mfcc'] = self._extract_mfcc(audio)
        
        # Mel-Spectrogram
        if self.mel_config.get('enabled', True):
            features['mel_spectrogram'] = self._extract_mel_spectrogram(audio)
        
        # Raw Waveform
        if self.config.get('model.features.raw_waveform.enabled', False):
            features['raw_waveform'] = audio
        
        # Spectral Features
        if self.spectral_config.get('enabled', True):
            spectral_features = self._extract_spectral_features(audio)
            features.update(spectral_features)
        
        return features
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        n_mfcc = self.mfcc_config.get('n_mfcc', 40)
        n_fft = self.mfcc_config.get('n_fft', 2048)
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=self.hop_length
        )
        
        # Add delta and delta-delta features for better temporal modeling
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack all features
        mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        return mfcc_features
    
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract Mel-Spectrogram."""
        n_mels = self.mel_config.get('n_mels', 128)
        n_fft = self.mel_config.get('n_fft', 2048)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features."""
        features = {}
        
        feature_names = self.spectral_config.get('features', [])
        
        if 'spectral_centroid' in feature_names:
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=audio, 
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
        
        if 'spectral_rolloff' in feature_names:
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
        
        if 'zero_crossing_rate' in feature_names:
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
                audio,
                hop_length=self.hop_length
            )
        
        if 'spectral_bandwidth' in feature_names:
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
        
        return features
    
    def _fix_length(self, audio: np.ndarray) -> np.ndarray:
        """Fix audio length to target duration."""
        target_length = int(self.duration * self.sample_rate)
        
        if len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Truncate
            audio = audio[:target_length]
        
        return audio
    
    def _remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove silent portions from audio.
        Uses a simple energy-based approach.
        """
        # Compute short-time energy
        frame_length = 2048
        energy = np.array([
            np.sum(np.abs(audio[i:i+frame_length]**2))
            for i in range(0, len(audio), frame_length)
        ])
        
        # Normalize energy
        energy = energy / np.max(energy) if np.max(energy) > 0 else energy
        
        # Identify non-silent frames
        non_silent = energy > self.silence_threshold
        
        # Expand frame indices to sample indices
        non_silent_samples = np.repeat(non_silent, frame_length)[:len(audio)]
        
        # Keep only non-silent audio
        audio_clean = audio[non_silent_samples]
        
        # If too much was removed, return original
        if len(audio_clean) < 0.1 * len(audio):
            logger.warning("Too much silence removed, returning original audio")
            return audio
        
        return audio_clean
    
    def _apply_preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis filter to amplify high frequencies.
        Helps in detecting subtle artifacts in synthetic speech.
        """
        return np.append(audio[0], audio[1:] - self.preemphasis_coef * audio[:-1])
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation techniques.
        Used during training only.
        
        Techniques:
        - Add random noise
        - Time stretching
        - Pitch shifting
        """
        aug_config = self.config.get('training.augmentation', {})
        
        if not aug_config.get('enabled', False):
            return audio
        
        # Add noise
        noise_factor = aug_config.get('noise_factor', 0.005)
        noise = np.random.randn(len(audio))
        audio = audio + noise_factor * noise
        
        # Time stretch (randomly)
        stretch_rate = np.random.uniform(*aug_config.get('time_stretch_rate', [0.9, 1.1]))
        audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
        
        # Pitch shift (randomly)
        pitch_steps = np.random.randint(*aug_config.get('pitch_shift_steps', [-2, 2]))
        audio = librosa.effects.pitch_shift(
            audio, 
            sr=self.sample_rate, 
            n_steps=pitch_steps
        )
        
        # Fix length again after augmentation
        audio = self._fix_length(audio)
        
        return audio


class FeatureNormalizer:
    """Normalize features for consistent model input."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, features: np.ndarray):
        """Compute mean and std from training data."""
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalize features."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        return (features - self.mean) / self.std
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)


if __name__ == "__main__":
    # Test preprocessing
    from utils.config import get_config
    
    config = get_config()
    preprocessor = AudioPreprocessor(config)
    
    print("AudioPreprocessor initialized successfully")
    print(f"Sample rate: {preprocessor.sample_rate} Hz")
    print(f"Duration: {preprocessor.duration} seconds")
    print(f"MFCC dimensions: {preprocessor.mfcc_config.get('n_mfcc')}")
