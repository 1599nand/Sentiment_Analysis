# Sentiment Analysis using Deep Learning

## Overview
This project is a **Sentiment Analysis Model** that classifies text reviews as **Positive or Negative** using **Deep Learning (BiLSTM)**. The model is trained on a dataset of customer reviews and converts text into numerical format using **word embeddings**.

## Features
✅ **Text Preprocessing:** Tokenization, stopword removal, and text vectorization using embeddings.  
✅ **Deep Learning Model:** Uses **Bidirectional LSTM (BiLSTM)** for text classification.  
✅ **Accuracy:** Achieves around **70%+ accuracy** on the test dataset.  
✅ **Balanced Data:** Handles imbalanced data using label balancing techniques.  
✅ **Scalability:** Can be integrated into APIs for real-time sentiment analysis.

## Dataset
The dataset consists of customer reviews with ratings ranging from 1 to 5. Reviews are converted into binary labels:
- **Positive (1):** Ratings of **4 and 5**
- **Negative (0):** Ratings of **1, 2, and 3**

## Model Architecture
- **Embedding Layer**: Converts text into numerical format.
- **Bidirectional LSTM**: Captures context in both forward and backward directions.
- **Dropout Layers**: Prevents overfitting.
- **Dense Layers**: Outputs the final classification result.

## Training Process
1. **Data Preprocessing**
2. **Tokenization & Embedding using Keras**
3. **Model Training (BiLSTM)**
4. **Evaluation (Accuracy, Precision, Recall)**

## Performance Metrics
- **Model Accuracy:** ~70%
- **Balanced label distribution for fair predictions**

## Installation
### Prerequisites
Make sure you have Python installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
Run the following command to train the model:
```bash
python sentiment_model.py
```

### Predicting Sentiment
(Optional) If deployed as an API, send a request:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "This product is amazing!"}'
```

## Deployment
The model can be deployed using **Flask/FastAPI** and hosted on platforms like **Render, Heroku, or Hugging Face Spaces**.

## Future Improvements
🔹 Fine-tune embeddings with pre-trained models (BERT, Word2Vec).  
🔹 Increase dataset size for better generalization.  
🔹 Deploy as a real-time sentiment analysis service.

## Author
👤 **Nand Nandan**  
📧 Contact: nandnandan.nn@gmail.com  
🔗 LinkedIn: www.linkedin.com/in/nand-nandan-1a99401b7  

---
🚀 **If you find this project useful, feel free to ⭐ the repo!**

