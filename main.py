import tensorflow as tf
import numpy as np

# Define a custom KAN layer
class KANLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, num_control_points=10):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_control_points = num_control_points
        
        # Initialize parameters for univariate functions (spline coefficients)
        self.weights = self.add_weight(
            name='weights',
            shape=(input_dim, output_dim, num_control_points),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Fixed control points for splines
        self.control_points = tf.linspace(-1.0, 1.0, num_control_points)

    def call(self, inputs):
        # inputs: (batch_size, input_dim)
        batch_size = tf.shape(inputs)[0]
        
        # Expand inputs for each output dimension
        inputs_expanded = tf.expand_dims(inputs, axis=-1)  # (batch_size, input_dim, 1)
        inputs_expanded = tf.tile(inputs_expanded, [1, 1, self.output_dim])  # (batch_size, input_dim, output_dim)
        
        # Simplified univariate function evaluation (linear approximation)
        # For full B-splines, extend with TensorFlow-compatible spline logic
        basis_functions = tf.reduce_sum(
            self.weights * tf.expand_dims(self.control_points, axis=0), axis=-1
        )  # (input_dim, output_dim)
        
        # Apply univariate functions and sum
        outputs = tf.einsum('bi,io->bo', inputs, basis_functions)  # (batch_size, output_dim)
        
        return outputs

# Build a KAN model
def build_kan_model(input_dim, hidden_dims, output_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    
    # Add KAN layers
    for hidden_dim in hidden_dims:
        x = KANLayer(input_dim, hidden_dim)(x)
        input_dim = hidden_dim
    
    # Output layer
    outputs = KANLayer(input_dim, output_dim)(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Generate synthetic data
def generate_synthetic_data(num_samples=1000):
    X = np.random.uniform(-1, 1, (num_samples, 2))
    y = np.sin(np.pi * X[:, 0]) + np.cos(np.pi * X[:, 1])  # Target function
    return X, y.reshape(-1, 1)

# Create and train the model
X, y = generate_synthetic_data()
model = build_kan_model(input_dim=2, hidden_dims=[10, 10], output_dim=1)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse',
              metrics=['mae'])

# Train the model
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X, y)
print(f"Test loss: {loss:.4f}, MAE: {mae:.4f}")
