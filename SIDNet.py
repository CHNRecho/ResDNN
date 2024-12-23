# -*- coding: utf-8 -*-
"""

@author: DELL
"""


import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    ReLU,
    Add
)
from tensorflow.keras.models import Model
from typing import Tuple, Optional


def build_dense_block(
    x: tf.Tensor,
    units: int,
    activation: Optional[tf.keras.layers.Layer] = ReLU(),
    kernel_initializer: str = 'he_normal',
    bias_initializer: str = 'zeros',
    use_batch_norm: bool = True,
    name: Optional[str] = None
) -> tf.Tensor:
    """
    Constructs a dense block consisting of a Dense layer, optional BatchNormalization, and an activation function.

    Args:
        x (tf.Tensor): Input tensor.
        units (int): Number of neurons in the Dense layer.
        activation (Optional[tf.keras.layers.Layer], optional): Activation function layer. Defaults to ReLU().
        kernel_initializer (str, optional): Kernel initializer for the Dense layer. Defaults to 'he_normal'.
        bias_initializer (str, optional): Bias initializer for the Dense layer. Defaults to 'zeros'.
        use_batch_norm (bool, optional): Whether to include BatchNormalization. Defaults to True.
        name (Optional[str], optional): Name for the Dense layer block. Defaults to None.

    Returns:
        tf.Tensor: Output tensor after applying Dense, BatchNormalization, and activation.
    """
    dense = Dense(
        units=units,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        use_bias=True,
        name=f"{name}_Dense" if name else None
    )(x)
    
    if use_batch_norm:
        dense = BatchNormalization(name=f"{name}_BatchNorm" if name else None)(dense)
    
    if activation is not None:
        dense = activation(dense)
    
    return dense


def SIDNet(
    input_size: Tuple[int] = (10,),
    units: int = 90,
    output_units: int = 2,
    learning_rate: float = 1e-4
) -> tf.keras.Model:
    """
    Builds the SIDNet model.

    Args:
        input_size (Tuple[int], optional): Shape of the input tensor. Defaults to (10,).
        units (int, optional): Number of neurons in each hidden Dense layer. Defaults to 90.
        output_units (int, optional): Number of neurons in the output layer. Defaults to 2.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-4.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    inputs = Input(shape=input_size, name='Input_Layer')

    # First Dense block
    x = build_dense_block(inputs, units, activation=ReLU(), name='Dense_1')

    # Second Dense block
    x = build_dense_block(x, units, activation=ReLU(), name='Dense_2')

    # Third Dense layer with residual connection
    dense3 = Dense(
        units=units,
        kernel_initializer='he_normal',
        bias_initializer='zeros',
        use_bias=True,
        name='Dense_3'
    )(x)
    bn3 = BatchNormalization(name='BatchNorm_3')(dense3)
    residual = Add(name='Add_Residual')([x, bn3])
    x = ReLU(name='ReLU_Residual')(residual)

    # Fourth Dense block
    x = build_dense_block(x, units, activation=ReLU(), name='Dense_4')

    # Output layer
    outputs = Dense(
        units=output_units,
        activation='linear',
        name='Output_Layer'
    )(x)

    # Build the model
    model = Model(inputs=inputs, outputs=outputs, name='SIDNet_Model')

    # Print model summary
    model.summary()

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='mae',
        optimizer=optimizer,
        metrics=['mse']
    )

    return model


if __name__ == "__main__":
    # Example: Create and view the model
    model = SIDNet()
