# Text-Emotion-Classifer
EmotiSense AI: A premium Streamlit dashboard for real-time text emotion detection, powered by Machine Learning (SVM) and Deep Learning (Bi-LSTM). 

# EmotiSense AI - Intelligent Emotion Analysis 🧠✨

EmotiSense is a web application that detects the underlying human emotion from text instantly. It provides an intuitive, glowing dark-theme dashboard to classify sentences into one of six distinct emotions: **Anger, Fear, Joy, Love, Sadness, and Surprise**.

The project uses Natural Language Processing (NLP) and offers two different models that you can switch between seamlessly:
- ⚡ **Machine Learning Model (Fast):** Uses `Linear SVM` combined with `TF-IDF` vectorization for lightning-fast inference.
- 🧠 **Deep Learning Model (Accurate):** Uses a `Bidirectional LSTM` neural network for high-accuracy contextual understanding.

## Features
- **Real-time Emotion Detection:** Instantly parses input text and predicts the dominant emotion.
- **Model Toggling:** Compare predictions between traditional Machine Learning and Deep Learning architectures with a single click.
- **Confidence Metrics:** Visualizes the probability distribution of all six emotions using horizontal progress bars.
- **Premium UI:** Built with Streamlit, featuring a fully custom CSS layout, glowing cards, and a responsive dashboard interface.

## Dataset
This project was trained on the [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) from Kaggle. 

## Tech Stack
- **Frontend:** Streamlit, Custom HTML/CSS
- **Machine Learning:** Scikit-Learn (LinearSVC, TF-IDF)
- **Deep Learning:** TensorFlow / Keras (Bi-LSTM, Embedding layers)
- **NLP Processing:** NLTK (Stopword removal, Stemming)
