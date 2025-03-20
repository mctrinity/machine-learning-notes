# TensorFlow Cheat Sheet

## **1. Tensor Operations**

### Creating Tensors:

```python
# Immutable tensor
 tf.constant(value, dtype=None)
# Mutable tensor (for training)
 tf.Variable(initial_value)
# Convert NumPy array to TensorFlow tensor
 tf.convert_to_tensor(numpy_array)
# Tensor filled with ones
 tf.ones(shape)
# Tensor filled with zeros
 tf.zeros(shape)
# Identity matrix
 tf.eye(size)
# Create a range of values
 tf.range(start, limit, delta)
# Random normal distribution
 tf.random.normal(shape, mean, stddev)
# Random uniform distribution
 tf.random.uniform(shape, minval, maxval)
```

### Tensor Manipulation:

```python
# Get shape of a tensor
 tf.shape(tensor)
# Reshape tensor
 tf.reshape(tensor, new_shape)
# Change tensor data type
 tf.cast(tensor, dtype)
# Add a new dimension
 tf.expand_dims(tensor, axis)
# Remove dimensions of size 1
 tf.squeeze(tensor)
# Transpose a tensor
 tf.transpose(tensor, perm)
```

### Mathematical Operations:

```python
# Addition
 tf.add(x, y)
# Subtraction
 tf.subtract(x, y)
# Element-wise multiplication
 tf.multiply(x, y)
# Matrix multiplication
 tf.matmul(A, B)
# Division
 tf.divide(x, y)
# Power
 tf.pow(x, y)
# Square root
 tf.sqrt(x)
# Sum across dimensions
 tf.reduce_sum(tensor, axis)
# Mean across dimensions
 tf.reduce_mean(tensor, axis)
# Max value
 tf.reduce_max(tensor)
# Min value
 tf.reduce_min(tensor)
```

---

## **2. Gradient & Auto-Differentiation**

```python
with tf.GradientTape() as tape:
    y = x ** 2  # Some function

# Compute dy/dx
 grad = tape.gradient(y, x)
```

---

## **3. Machine Learning - Neural Networks**

### Building a Model:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units, activation='relu'),
    tf.keras.layers.Dense(output_units)
])
```

### Compiling & Training:

```python
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10, batch_size=32)
model.evaluate(test_data, test_labels)
```

### Predicting:

```python
predictions = model.predict(new_data)
```

---

## **4. TensorFlow Data Handling**

### NumPy Conversion:

```python
# Convert TensorFlow tensor to NumPy array
 tensor.numpy()
# Convert NumPy to TensorFlow tensor
 tf.convert_to_tensor(numpy_array)
```

### Data Pipelines:

```python
# Create dataset from data
 dataset = tf.data.Dataset.from_tensor_slices(data)
# Shuffle, batch, and prefetch
 dataset = dataset.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)
```

---

## **5. GPU Acceleration**

```python
# Check if GPU is available
 print(tf.config.list_physical_devices('GPU'))
```

---

## **6. Saving & Loading Models**

```python
# Save model
 model.save("my_model.h5")
# Load model
 model = tf.keras.models.load_model("my_model.h5")
```

---

## **7. TensorFlow Keras Callbacks**

```python
# Stop early if no improvement
 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Save best model
 tf.keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True)
```
