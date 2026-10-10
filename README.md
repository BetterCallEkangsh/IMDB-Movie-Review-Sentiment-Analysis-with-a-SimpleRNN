# 🎬 ReelSentiment: IMDB Movie Review Sentiment Analysis with a SimpleRNN

A deep learning project that classifies movie reviews as **Positive** or **Negative** using a **Simple Recurrent Neural Network (RNN)** built with TensorFlow/Keras, served through an interactive **Streamlit** web app.

---

## 📌 Overview

This project walks through the full NLP pipeline, from understanding word embeddings to deploying a trained RNN:

1. **Text to numbers**: reviews are converted into integer sequences using the IMDB word index.
2. **Embedding**: each word index is mapped to a dense 128-dimensional vector.
3. **SimpleRNN**: reads the review word by word, carrying a hidden state forward through time.
4. **Sigmoid output**: produces a score between 0 and 1 (above 0.5 means positive).
5. **Streamlit app**: type any review and get a prediction instantly.

---

## 🧠 Model Architecture

| Layer | Output Shape | Params |
|---|---|---|
| Embedding (vocab 10,000, dim 128) | (None, 500, 128) | 1,280,000 |
| SimpleRNN (128 units, ReLU) | (None, 128) | 32,896 |
| Dense (1 unit, sigmoid) | (None, 1) | 129 |

**Total parameters:** 1,313,025 (~5 MB)

- **Dataset:** Keras IMDB movie reviews (binary sentiment)
- **Vocabulary size:** 10,000 most frequent words
- **Max review length:** 500 tokens (padded or truncated)

---

## 📁 Project Structure

```
├── embedding.ipynb        # One-hot indexing, padding, and Embedding layer basics
├── simplernn.ipynb        # Building and training the SimpleRNN on IMDB
├── prediction.ipynb       # Loading the trained model and predicting on new reviews
├── main.py                # Streamlit web app
├── simple_rnn_imdb.h5     # Trained model weights
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/BetterCallEkangsh/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Tested with Python 3.11 and TensorFlow 2.15.0.

### 4. Run the app

```bash
streamlit run main.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 💡 Usage

1. Type or paste a movie review into the text box.
2. Click **Classify**.
3. The app shows the predicted **sentiment** and the raw **prediction score**.

**Example**

```
Review:  This movie was fantastic! The acting was great and the plot was thrilling.
Sentiment: Positive
Prediction Score: 0.81
```

---

## ⚙️ How Preprocessing Works

```python
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded_review], maxlen=500)
```

- Words are lowercased and split on spaces.
- Each word is looked up in the IMDB word index. Unknown words map to index 2.
- Indices are shifted by 3 because Keras reserves 0 (padding), 1 (start), and 2 (unknown).
- The sequence is padded to a fixed length of 500.

---

## 📓 Notebooks

| Notebook | What it covers |
|---|---|
| `embedding.ipynb` | Converting sentences to indices, `pad_sequences`, and how the Embedding layer turns indices into dense vectors |
| `simplernn.ipynb` | Loading IMDB, building the SimpleRNN model, training, and saving it |
| `prediction.ipynb` | Loading the saved `.h5` model, preprocessing raw text, and predicting sentiment |

---

## 🔭 Possible Improvements

- Replace `SimpleRNN` with **LSTM** or **GRU** to handle long-range dependencies better.
- Use **pretrained embeddings** (GloVe or Word2Vec) instead of learning them from scratch.
- Strip punctuation in `preprocess_text` so words like `fantastic!` match the vocabulary.
- Add **EarlyStopping** and hyperparameter tuning.
- Deploy the app on Streamlit Community Cloud or Hugging Face Spaces.

---

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `NumPy` · `Streamlit` · `Jupyter`

---

## 👤 Author

**Ekangsh**
GitHub: [@BetterCallEkangsh](https://github.com/BetterCallEkangsh)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).