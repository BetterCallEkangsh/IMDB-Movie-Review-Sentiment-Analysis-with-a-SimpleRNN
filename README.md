# 🎬 ReelSentiment: IMDB Movie Review Sentiment Analysis with a SimpleRNN

A deep learning project that classifies movie reviews as **Positive** or **Negative** using a **Simple Recurrent Neural Network (RNN)** built with TensorFlow/Keras, served through an interactive **Streamlit** web app.

---

## 📌 Overview

The project covers the full pipeline, from raw text to a working app:

1. **Text to numbers**: reviews are cleaned and converted into integer sequences using the IMDB word index.
2. **Embedding**: each word index is mapped to a dense 128-dimensional vector.
3. **SimpleRNN**: reads the review word by word, carrying a hidden state forward through time.
4. **Dense layers + sigmoid**: produce a score between 0 and 1 (above 0.5 means positive).
5. **Streamlit app**: type any review and get a prediction instantly.

---

## 🧠 Model Architecture

| Layer | Output Shape | Params |
|---|---|---|
| Embedding (vocab 10,000, dim 128) | (batch, 500, 128) | 1,280,000 |
| SimpleRNN (128 units) | (batch, 128) | 32,896 |
| Dense (128 units) | (batch, 128) | 16,512 |
| Dense (1 unit, sigmoid) | (batch, 1) | 129 |

**Total trainable parameters:** 1,329,537 (~5 MB)

- **Dataset:** Keras IMDB movie reviews (25,000 train / 25,000 test, binary sentiment)
- **Vocabulary size:** 10,000 most frequent words
- **Max review length:** 500 tokens (padded or truncated)
- **Test accuracy:** `XX.XX%` *(replace with your `model.evaluate` result)*

---

## 📁 Project Structure

```
├── experiments.ipynb      # Data loading, padding, model building, training, saving
├── predictions.ipynb      # Load the saved model and predict sentiment on new reviews
├── main.py                # Streamlit web app
├── IMDB_Sentiment.h5      # Trained model
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

### 2. Create and activate a virtual environment

Using `venv`:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

Or using conda:

```bash
conda create -n rnn_env python=3.11
conda activate rnn_env
```

> Developed with Python 3.11.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

Run it from the project folder, so the model file can be found:

```bash
streamlit run main.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 💡 Usage

1. Type or paste a movie review into the text box.
2. Click **Classify**.
3. The app shows the predicted **sentiment** and the raw **prediction score**.

**Sample predictions**

| Review | Prediction | Score |
|---|---|---|
| This movie was good. I really liked it | Positive | 0.91 |
| This movie was fantastic! The acting was great and the plot was thrilling. | Positive | 0.87 |
| One of the best films I have seen, I loved every minute of it. | Positive | 0.97 |
| Absolutely terrible. Boring plot and awful acting, a waste of time. | Negative | 0.001 |

---

## ⚙️ How Preprocessing Works

```python
def preprocess_text(text, vocab_size=10000):
    words = re.sub(r"[^\w\s']", "", text.lower()).split()
    encoded = []
    for w in words:
        i = word_index.get(w)
        encoded.append(i + 3 if i is not None and i + 3 < vocab_size else 2)
    return sequence.pad_sequences([encoded], maxlen=500)
```

- Text is lowercased and **punctuation is removed**, so `fantastic!` matches the word `fantastic` in the vocabulary.
- Each word is looked up in the IMDB word index. Keras shifts every index by 3, because 0 (padding), 1 (start) and 2 (unknown) are reserved.
- Words that are not in the vocabulary, or fall outside the top 10,000, are mapped to **2 (unknown)**.
- The sequence is padded to a fixed length of 500.

---

## ⚠️ Limitations

- A **SimpleRNN** forgets earlier words over long sequences (the vanishing gradient problem), so predictions can be unreliable on very short or mixed-sentiment reviews.
- The model only knows the 10,000 most frequent IMDB words, and it was trained on movie reviews, so it may not transfer well to other domains.

---

## 🔭 Possible Improvements

- Replace `SimpleRNN` with **LSTM** or **GRU** and compare accuracy.
- Use **pretrained embeddings** (GloVe or Word2Vec) instead of learning them from scratch.
- Add **EarlyStopping** and tune hyperparameters (units, embedding size, dropout).
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