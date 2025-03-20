# TensorFlow Cheat Sheet

## **1. Tensor Operations**

### Creating Tensors:

```python
# Immutable tensor (used for constants, values cannot be changed after creation)
tf.constant(value, dtype=None)

# Mutable tensor (used for variables whose values change during training)
tf.Variable(initial_value)

# Convert NumPy array to TensorFlow tensor (useful when integrating NumPy and TensorFlow)
tf.convert_to_tensor(numpy_array)

# Tensor filled with ones (commonly used for initializing weights or biases)
tf.ones(shape)

# Tensor filled with zeros (often used for padding or initialization)
tf.zeros(shape)

# Identity matrix (useful in linear algebra and deep learning models)
tf.eye(size)

# Create a range of values (similar to Python's range function, useful for indexing)
tf.range(start, limit, delta)

# Random normal distribution (used for initializing neural network weights)
tf.random.normal(shape, mean, stddev)

# Random uniform distribution (another way to initialize values in ML models)
tf.random.uniform(shape, minval, maxval)
```

### Tensor Manipulation:

```python
# Get shape of a tensor (helps in debugging and understanding data dimensions)
tf.shape(tensor)

# Reshape tensor (commonly used for feeding data into ML models in the correct format)
tf.reshape(tensor, new_shape)

# Change tensor data type (useful for precision control and compatibility)
tf.cast(tensor, dtype)

# Add a new dimension (useful when preparing data for ML models)
tf.expand_dims(tensor, axis)

# Remove dimensions of size 1 (helps in reducing unnecessary dimensions)
tf.squeeze(tensor)

# Transpose a tensor (commonly used in matrix operations and CNNs)
tf.transpose(tensor, perm)
```

### Mathematical Operations:

```python
# Addition of tensors (element-wise)
tf.add(x, y)

# Subtraction (element-wise)
tf.subtract(x, y)

# Element-wise multiplication (useful in applying operations on tensors)
tf.multiply(x, y)

# Matrix multiplication (fundamental in neural networks and deep learning)
tf.matmul(A, B)

# Division (element-wise)
tf.divide(x, y)

# Power (raising each element to a power)
tf.pow(x, y)

# Square root of elements
tf.sqrt(x)

# Sum across dimensions (used to reduce tensor dimensions in operations)
tf.reduce_sum(tensor, axis)

# Mean across dimensions (useful for normalization and averaging operations)
tf.reduce_mean(tensor, axis)

# Maximum value in a tensor
tf.reduce_max(tensor)

# Minimum value in a tensor
tf.reduce_min(tensor)
```

---

## **2. Gradient & Auto-Differentiation**

```python
# TensorFlow automatically tracks operations on tensors
with tf.GradientTape() as tape:
    y = x ** 2  # Some function

# Compute dy/dx (used in backpropagation and optimization)
grad = tape.gradient(y, x)
```

---

## **3. Machine Learning - Neural Networks**

### Building a Model:

```python
# Define a basic neural network with layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units, activation='relu'),  # Hidden layer with ReLU activation
    tf.keras.layers.Dense(output_units)  # Output layer (e.g., classification or regression)
])
```

### Compiling & Training:

```python
# Compile model with optimizer and loss function (Adam optimizer is commonly used)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train model with input data
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# Evaluate model performance on test data
model.evaluate(test_data, test_labels)
```

### Predicting:

```python
# Make predictions using trained model
predictions = model.predict(new_data)
```

---

## **4. TensorFlow Data Handling**

### NumPy Conversion:

```python
# Convert TensorFlow tensor to NumPy array (useful for visualization and debugging)
tensor.numpy()

# Convert NumPy to TensorFlow tensor (for ML model compatibility)
tf.convert_to_tensor(numpy_array)
```

### Data Pipelines:

```python
# Create dataset from raw data (efficient data loading)
dataset = tf.data.Dataset.from_tensor_slices(data)

# Shuffle, batch, and prefetch (improves training speed and efficiency)
dataset = dataset.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)
```

---

## **5. GPU Acceleration**

```python
# Check if GPU is available for faster computations
print(tf.config.list_physical_devices('GPU'))
```

---

## **6. Saving & Loading Models**

```python
# Save model to disk (useful for resuming training later)
model.save("my_model.h5")

# Load saved model for inference or further training
model = tf.keras.models.load_model("my_model.h5")
```

---

## **7. TensorFlow Keras Callbacks**

```python
# Stop training early if validation loss does not improve
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Save the best model version during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True)
```
