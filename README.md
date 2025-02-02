# 🎯 AI-Based Sentiment Analyzer  
📝 **Pretrained LSTM Model for Sentiment Classification using NLP**  

## 🚀 Overview  
This AI-based sentiment analyzer classifies text as **positive, negative, or neutral** using a **pretrained LSTM (Long Short-Term Memory) model** from **TensorFlow Hub**. The project leverages **natural language processing (NLP) techniques** to preprocess and analyze text efficiently.

## 🏗️ Features  
✅ Uses a **pretrained LSTM model** (instead of training from scratch) for efficient sentiment classification.  
✅ **Text preprocessing pipeline** with **NLTK** for tokenization, stopword removal, and stemming.  
✅ Converts text into **numerical sequences** using **TensorFlow Tokenizer** and **word embeddings**.  
✅ **Dataset split**: 80% Training, 20% Testing using **scikit-learn**.  
✅ Model evaluation using **accuracy, precision, recall, and F1-score** metrics.  

## 🛠️ Tech Stack  
- **Programming Language:** Python  
- **Libraries Used:**  
  - 🏗️ `TensorFlow Hub` → Pretrained LSTM model for sentiment classification  
  - 🧠 `NLTK` → Text preprocessing (tokenization, stopword removal, stemming)  
  - 📊 `pandas` → Data manipulation & CSV handling  
  - 📈 `scikit-learn` → Dataset splitting & evaluation metrics  

---

## 📦 Requirements  

To run this project, you need the following:  

- **Python** (>= 3.8)  
- **Libraries:**  
  - `tensorflow` → For loading the pretrained LSTM model  
  - `tensorflow-hub` → To use TensorFlow Hub's sentiment model  
  - `nltk` → For text preprocessing (tokenization, stopword removal, stemming)  
  - `pandas` → For dataset handling and CSV processing  
  - `scikit-learn` → For dataset splitting and evaluation metrics  

Additionally, download NLTK stopwords and tokenizer data before running the project.  



