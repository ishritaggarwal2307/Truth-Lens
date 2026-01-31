"""
Truth-Lens Ensemble Model Architecture
=======================================
Advanced CNN with:
- Multi-feature input (MFCC, Mel-Spectrogram, Spectral)
- Attention mechanism
- Ensemble predictions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D,
    Attention, MultiHeadAttention, LayerNormalization
)
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """
    Custom attention layer for focusing on important temporal features.
    """
    
    def __init__(self, attention_dim=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
    
    def build(self, input_shape):
        """Build attention weights."""
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        """Apply attention mechanism."""
        # Compute attention scores
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        
        # Apply attention weights
        ait = tf.expand_dims(ait, axis=-1)
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_cnn_branch(input_shape, name_prefix, config):
    """
    Build a CNN branch for processing a single feature type.
    
    Args:
        input_shape: Shape of input features
        name_prefix: Prefix for layer names
        config: Model configuration
        
    Returns:
        Keras model
    """
    inputs = layers.Input(shape=input_shape, name=f"{name_prefix}_input")
    
    x = inputs
    
    # Add channel dimension if needed
    if len(input_shape) == 2:
        x = layers.Reshape((*input_shape, 1))(x)
    
    # Convolutional layers
    conv_layers = config.get('model.cnn.conv_layers', [])
    
    for idx, conv_config in enumerate(conv_layers):
        filters = conv_config['filters']
        kernel_size = tuple(conv_config['kernel_size'])
        activation = conv_config['activation']
        pool_size = tuple(conv_config.get('pool_size', [2, 2]))
        
        x = Conv2D(
            filters,
            kernel_size,
            activation=activation,
            padding='same',
            name=f"{name_prefix}_conv_{idx+1}"
        )(x)
        x = BatchNormalization(name=f"{name_prefix}_bn_{idx+1}")(x)
        x = MaxPooling2D(pool_size, name=f"{name_prefix}_pool_{idx+1}")(x)
    
    # Global pooling
    x = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    
    model = models.Model(inputs=inputs, outputs=x, name=f"{name_prefix}_branch")
    
    return model


def build_ensemble_model(config):
    """
    Build complete ensemble model with multiple feature branches.
    
    Args:
        config: Configuration object
        
    Returns:
        Compiled Keras model
    """
    feature_branches = []
    branch_outputs = []
    
    # MFCC Branch
    if config.get('model.features.mfcc.enabled', True):
        n_mfcc = config.get('model.features.mfcc.n_mfcc', 40)
        # With delta and delta-delta: 3 * n_mfcc
        mfcc_shape = (3 * n_mfcc, None)  # Variable time dimension
        
        # For fixed time dimension (based on 3 seconds)
        time_steps = int((config.get('audio.duration', 3.0) * 
                         config.get('audio.sample_rate', 16000)) / 
                        config.get('audio.hop_length', 512))
        mfcc_shape = (3 * n_mfcc, time_steps)
        
        mfcc_branch = build_cnn_branch(mfcc_shape, "mfcc", config)
        feature_branches.append(mfcc_branch)
        branch_outputs.append(mfcc_branch.output)
    
    # Mel-Spectrogram Branch
    if config.get('model.features.mel_spectrogram.enabled', True):
        n_mels = config.get('model.features.mel_spectrogram.n_mels', 128)
        time_steps = int((config.get('audio.duration', 3.0) * 
                         config.get('audio.sample_rate', 16000)) / 
                        config.get('audio.hop_length', 512))
        mel_shape = (n_mels, time_steps)
        
        mel_branch = build_cnn_branch(mel_shape, "mel", config)
        feature_branches.append(mel_branch)
        branch_outputs.append(mel_branch.output)
    
    # Concatenate all branch outputs
    if len(branch_outputs) > 1:
        concatenated = layers.Concatenate(name="feature_concat")(branch_outputs)
    else:
        concatenated = branch_outputs[0]
    
    # Attention mechanism
    if config.get('model.attention.enabled', True):
        attention_dim = config.get('model.attention.attention_dim', 128)
        
        # Reshape for attention (add sequence dimension)
        x = layers.Reshape((1, concatenated.shape[-1]))(concatenated)
        x = AttentionLayer(attention_dim, name="attention")(x)
    else:
        x = concatenated
    
    # Dense layers
    dense_layers = config.get('model.cnn.dense_layers', [])
    
    for idx, dense_config in enumerate(dense_layers):
        units = dense_config['units']
        activation = dense_config['activation']
        dropout = dense_config.get('dropout', 0.0)
        
        x = Dense(units, activation=activation, name=f"dense_{idx+1}")(x)
        x = BatchNormalization(name=f"dense_bn_{idx+1}")(x)
        
        if dropout > 0:
            x = Dropout(dropout, name=f"dropout_{idx+1}")(x)
    
    # Output layer
    output_units = config.get('model.cnn.output.units', 2)
    output_activation = config.get('model.cnn.output.activation', 'softmax')
    
    outputs = Dense(output_units, activation=output_activation, name="output")(x)
    
    # Create final model
    model_inputs = [branch.input for branch in feature_branches]
    model = models.Model(inputs=model_inputs, outputs=outputs, name="TruthLens_Ensemble")
    
    # Compile model
    optimizer_name = config.get('training.optimizer', 'adam')
    learning_rate = config.get('training.learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = optimizer_name
    
    loss = config.get('training.loss', 'categorical_crossentropy')
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    logger.info(f"Model built: {model.name}")
    logger.info(f"Total parameters: {model.count_params():,}")
    
    return model


def build_lightweight_model(config):
    """
    Build a lightweight single-branch model for faster inference.
    Use this for the hackathon demo if the ensemble is too slow.
    
    Args:
        config: Configuration object
        
    Returns:
        Compiled Keras model
    """
    n_mfcc = config.get('model.features.mfcc.n_mfcc', 40)
    time_steps = int((config.get('audio.duration', 3.0) * 
                     config.get('audio.sample_rate', 16000)) / 
                    config.get('audio.hop_length', 512))
    
    input_shape = (3 * n_mfcc, time_steps, 1)
    
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape, name="input"),
        
        # Conv Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1"),
        BatchNormalization(name="bn1"),
        MaxPooling2D((2, 2), name="pool1"),
        
        # Conv Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2"),
        BatchNormalization(name="bn2"),
        MaxPooling2D((2, 2), name="pool2"),
        
        # Conv Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3"),
        BatchNormalization(name="bn3"),
        MaxPooling2D((2, 2), name="pool3"),
        
        # Global pooling
        GlobalAveragePooling2D(name="gap"),
        
        # Dense layers
        Dense(256, activation='relu', name="dense1"),
        Dropout(0.5, name="dropout1"),
        
        Dense(128, activation='relu', name="dense2"),
        Dropout(0.3, name="dropout2"),
        
        # Output
        Dense(2, activation='softmax', name="output")
    ], name="TruthLens_Lightweight")
    
    # Compile
    optimizer = keras.optimizers.Adam(
        learning_rate=config.get('training.learning_rate', 0.001)
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    logger.info(f"Lightweight model built with {model.count_params():,} parameters")
    
    return model


def get_model_summary(model):
    """Get formatted model summary."""
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)


if __name__ == "__main__":
    # Test model building
    from utils.config import get_config
    
    config = get_config()
    
    print("Building ensemble model...")
    model = build_ensemble_model(config)
    print("\n" + get_model_summary(model))
    
    print("\n" + "="*80)
    print("Building lightweight model...")
    lightweight = build_lightweight_model(config)
    print("\n" + get_model_summary(lightweight))
