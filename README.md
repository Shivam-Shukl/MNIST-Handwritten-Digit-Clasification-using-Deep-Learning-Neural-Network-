# MNIST Handwritten Digit Classification 🖊️🔢

A deep learning project that classifies handwritten digits (0–9) from the **MNIST dataset** using a fully connected neural network built with TensorFlow/Keras.

---

## 📌 Project Overview

- Implemented a **feedforward neural network classifier** for MNIST digit recognition
- Preprocessed and normalized 28×28 grayscale images
- Visualized training data and model predictions
- Trained on 60,000 images and evaluated on 10,000 test images
- Achieved strong classification performance on unseen handwritten digits

---

## 🚀 Features

- ✅ Loads MNIST dataset via `keras.datasets.mnist`
- ✅ Data preprocessing: normalization and flattening
- ✅ Visualization tools using `matplotlib`, `seaborn`, and `OpenCV`
- ✅ Deep neural network with 2 hidden layers
- ✅ Model evaluation with accuracy metrics and confusion matrix
- ✅ Sample prediction visualization on test images

---

## 📂 Project Structure

```
.
├── MnistdigitClassification.ipynb   # Main notebook with full workflow
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

---

## ⚙️ Requirements

Install all dependencies with:

```bash
pip install numpy matplotlib seaborn opencv-python pillow tensorflow
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Required Libraries:**
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `seaborn` - Statistical plotting
- `opencv-python` (cv2) - Image processing
- `pillow` (PIL) - Image handling
- `tensorflow` - Deep learning framework (Keras API)

---

## 📊 Dataset

**MNIST Handwritten Digits Database**
- 📥 60,000 training images
- 📥 10,000 test images
- 📐 Image size: **28×28 pixels** (grayscale)
- 🏷️ Labels: 10 classes (digits 0–9)

The dataset is automatically downloaded via `keras.datasets.mnist.load_data()`.

---

## 🏗️ Model Architecture

**Type:** Fully Connected Neural Network (Feedforward)

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input: 28×28 → 784 features
    keras.layers.Dense(50, activation='relu'),   # Hidden layer 1: 50 neurons
    keras.layers.Dense(50, activation='relu'),   # Hidden layer 2: 50 neurons
    keras.layers.Dense(10, activation='sigmoid') # Output layer: 10 classes
])
```

**Model Configuration:**
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy
- **Training:** 10 epochs

**Layer Breakdown:**
1. **Input Layer:** Flattens 28×28 images into 784-dimensional vectors
2. **Hidden Layer 1:** 50 neurons with ReLU activation
3. **Hidden Layer 2:** 50 neurons with ReLU activation
4. **Output Layer:** 10 neurons with sigmoid activation (one per digit class)

---

## 📈 Results

**Training Configuration:**
- Epochs: 10
- Training samples: 60,000
- Test samples: 10,000

**Performance:**
- Training accuracy: ~97-98%
- Test accuracy: ~97%+
- Model generalizes well across all digit classes (0–9)

> **Note:** Exact accuracy values can be found in the notebook output cells.

---

## 📊 Visualizations

The notebook includes:
- 📸 Sample training images with labels
- 📉 Training loss and accuracy curves over epochs
- 🎯 Confusion matrix showing classification performance per digit
- ✅ Correctly classified vs. ❌ misclassified digit examples
- 🔮 Prediction visualizations on test set

---

## ▶️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd mnist-digit-classification
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch Jupyter Notebook

```bash
jupyter notebook MnistdigitClassification.ipynb
```

### 4️⃣ Run the Notebook

- Execute all cells sequentially (`Cell > Run All`)
- The model will train automatically and display results
- Visualizations will appear inline

---

## 🔍 Key Steps in the Notebook

1. **Data Loading:** Import MNIST dataset
2. **Exploration:** Check data shapes and visualize samples
3. **Preprocessing:** Normalize pixel values (0–1 range)
4. **Model Building:** Define neural network architecture
5. **Training:** Fit model on training data
6. **Evaluation:** Test on unseen data and calculate accuracy
7. **Visualization:** Plot predictions and confusion matrix

---

## 📌 Future Improvements

- 🧠 Implement **Convolutional Neural Networks (CNNs)** for improved accuracy (>99%)
- 🔄 Add **data augmentation** (rotation, shifting) for robustness
- 📊 Experiment with different architectures (deeper networks, dropout layers)
- 🎨 Create an interactive **digit drawing interface** for real-time predictions
- 🚀 Deploy model as a web app using Flask/Streamlit
- 🔧 Hyperparameter tuning (learning rate, batch size, optimizer)

---

## 📝 License

This project is licensed under the **MIT License** - feel free to use and modify.

---

## 👨‍💻 Author

Developed as a deep learning practice project for handwritten digit recognition.

---

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- TensorFlow/Keras documentation and tutorials
- Deep learning community resources

---

**⭐ If you found this project helpful, consider giving it a star!**
