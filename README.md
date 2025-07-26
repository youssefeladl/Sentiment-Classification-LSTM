# Sentiment Classification using LSTM

## 🧠 Project Overview

This project focuses on classifying the sentiment of movie reviews (positive or negative) using Natural Language Processing (NLP) techniques combined with a deep learning model based on LSTM (Long Short-Term Memory) networks.

The model was trained on a dataset of 1,999 labeled movie reviews and evaluated on both traditional vectorization techniques (TF-IDF, N-Gram) and learned word embeddings.

---

## 🗂️ Dataset Summary

- **Total Samples**: 1,999 reviews  
- **Sentiment Labels**: `positive`, `negative`  
- **Unique Reviews**: 1,999  
- **Label Distribution**:  
  - Positive: 1005  
  - Negative: 994  

---

## ✨ Feature Extraction

Multiple feature representations were explored:

- **TF-IDF Vectors**  
  - Shape: `(1999, 17155)`  
  - Captures term importance relative to the corpus  

- **N-Gram Vectors**  
  - Shape: `(1999, 177377)`  
  - Captures sequences of words and contextual patterns  

- **Learned Word Embeddings**  
  - Shape: `(1999, 384)`  
  - Generated using an embedding layer trained with the model  

---

## 🧠 Model Architecture

The final model used is a deep neural network based on LSTM layers. Here is a summary:

- **Embedding Layer**: Converts words into 128-dimensional vectors  
- **Bidirectional LSTM**: Captures forward and backward dependencies  
- **Dropout Layer**: Reduces overfitting (rate = 0.5)  
- **Dense Output Layer**: Sigmoid activation for binary classification

### 🔧 Model Summary

| Layer              | Output Shape     | Parameters |
|--------------------|------------------|------------|
| Embedding          | (None, 500, 128) | 2,198,784  |
| Bidirectional LSTM | (None, 128)      | 98,816     |
| Dropout            | (None, 128)      | 0          |
| Dense              | (None, 1)        | 129        |
| **Total**          |                  | **2,297,729** trainable params |

---

## 🏋️ Training Details

- **Training samples**: 1,199  
- **Testing samples**: 800  
- **Sequence length**: 500 tokens  
- **Epochs**: 15  
- **Optimizer**: Adam  
- **Loss Function**: Binary Crossentropy  
- **Learning Rate Schedule**: ReduceLROnPlateau applied

#### 📈 Epoch Highlights

| Epoch | Accuracy | Val Accuracy | Val Loss | LR       |
|-------|----------|--------------|----------|----------|
| 1     | 89.1%    | 77.3%        | 0.4833   | 0.0010   |
| 2     | 96.2%    | 80.2%        | 0.4788   | 0.0010   |
| 3     | 98.7%    | 80.6%        | 0.5250   | 0.0010   |
| 4     | 99.7%    | 80.8%        | 0.5945   | 0.0002   |
| 5     | 99.8%    | 81.5%        | 0.5929   | 0.0002   |

---

## 📊 Evaluation Metrics

Final model evaluation on the test set (800 samples):

- ✅ **Test Accuracy**: 84%
- 🎯 **Precision**: 84%
- 📥 **Recall**: 85%
- 📊 **F1 Score**: 0.85

---

## 🔮 Future Work

- Use pretrained embeddings (e.g. GloVe, Word2Vec)
- Apply advanced models like BERT
- Deploy as an API or build a web app for real-time prediction

---

## ⚙️ Tools & Libraries

- Python 3.11
- Keras / TensorFlow
- Pandas, NumPy, Scikit-learn
- NLTK
- Matplotlib / Seaborn (for visualizations, optional)

---

## 📌 Notes

- The `input_length` argument in Keras `Embedding` layer is now deprecated.
- All models were trained on local machine (CPU/GPU environment).
