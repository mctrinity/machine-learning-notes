# Machine Learning Workflow with TensorFlow

## **1️⃣ Data Digestion (Raw Data Ingestion)**
### **Goal:** Load data from various sources (files, APIs, databases).
### **Tasks:**
- Collect structured (CSV, JSON) or unstructured data (images, text, audio).
- Split data into **training** and **testing** sets.
- Inspect data structure (rows, columns, labels, image dimensions).

### **Tools Used:**
- **Pandas / NumPy** → For structured data (CSV, JSON)
- **OpenCV / PIL** → For image processing
- **TensorFlow (tf.data)** → For handling large datasets

✅ **TensorFlow is used minimally in this stage, mainly for large dataset handling.**

---

## **2️⃣ Data Preprocessing (Data Hygiene & Transformation)**
### **Goal:** Clean and prepare the data for training.
### **Tasks:**
- **Normalization** → Scale numbers (e.g., pixel values 0-255 → 0-1)
- **Reshaping** → Convert data to correct input format
- **Encoding Labels** → Convert categorical labels into numerical form
- **Handling Missing Data** → Fill or remove missing values

### **Tools Used:**
- **NumPy / Pandas** → For data transformations
- **Scikit-learn** → For encoding labels, feature scaling
- **TensorFlow (tf.data.Dataset.map())** → For efficient pipelines

✅ **TensorFlow helps here but is not the primary tool.**

---

## **3️⃣ Model Building (Defining the Neural Network)**
### **Goal:** Create the neural network architecture.
### **Tasks:**
- Define layers (Input, Hidden, Output)
- Choose activation functions (ReLU, Softmax, etc.)
- Compile the model (optimizer, loss function, and metrics)

### **Tools Used:**
- ✅ **TensorFlow (Keras API)** → `keras.Sequential([...])`

✅ **TensorFlow is the primary tool in this stage.**

---

## **4️⃣ Model Training (Learning from Data)**
### **Goal:** Train the model by adjusting weights using data.
### **Tasks:**
- Feed input data through the model
- Compute loss/error
- Adjust weights using **gradient descent (backpropagation)**
- Iterate over multiple **epochs**

### **Tools Used:**
- ✅ **TensorFlow (Keras API)** → `.fit()`

✅ **TensorFlow does all the heavy lifting here.**

---

## **5️⃣ Model Evaluation (Testing Performance)**
### **Goal:** Check how well the model performs on unseen data.
### **Tasks:**
- Evaluate accuracy, loss, and performance metrics
- Tune model hyperparameters if needed

### **Tools Used:**
- ✅ **TensorFlow (Keras API)** → `.evaluate()`

✅ **TensorFlow is fully responsible for this stage.**

---

## **6️⃣ Making Predictions (Using the Model in Real Life)**
### **Goal:** Use the trained model to make predictions on new data.
### **Tasks:**
- Provide new input data
- Convert raw model output into meaningful results

### **Tools Used:**
- ✅ **TensorFlow (Keras API)** → `.predict()`

✅ **TensorFlow takes input data and makes final predictions.**

---

## **Final Summary**
| **Stage**               | **Main Tools Used**              | **Does TensorFlow Help?** |
|----------------------|---------------------------------|--------------------------|
| **1️⃣ Data Digestion** | NumPy, Pandas, OpenCV, tf.data  | ✅ A little (dataset loading) |
| **2️⃣ Data Hygiene**   | NumPy, Pandas, scikit-learn     | ✅ Sometimes (normalization, reshaping) |
| **3️⃣ Model Building** | ✅ **TensorFlow** (Keras, Layers)  | ✅ Yes (Neural Network creation) |
| **4️⃣ Training**       | ✅ **TensorFlow (fit, backpropagation)** | ✅ 100% TensorFlow |
| **5️⃣ Evaluation**     | ✅ **TensorFlow (evaluate, metrics)** | ✅ 100% TensorFlow |
| **6️⃣ Predictions**    | ✅ **TensorFlow (predict, inference)** | ✅ 100% TensorFlow |

✅ **TensorFlow is mainly used for model building, training, evaluation, and predictions.**  
✅ **It can assist with preprocessing but is not the primary tool for that.**

