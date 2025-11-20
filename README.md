# Next Word Predictor for Gujarati Language 🕉️

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Gujarati-green)

## 📌 Project Overview
This project focuses on Natural Language Processing (NLP) for **Gujarati**, a regional Indic language. The goal is to predict the most probable next word in a sequence using Deep Learning techniques.

Due to the complex grammar rules of Indic languages and the lack of tagged datasets, this project utilizes a **Recurrent Neural Network (RNN)** approach—specifically **Gated Recurrent Units (GRU)**—trained on an untagged plain text corpus.

## 👥 Collaborators
This project is a collaborative effort by:
* **[Dharmil2684](https://github.com/Dharmil2684)**
* **[MitkumarR](https://github.com/MitkumarR)**

---

## 🚀 Features
* **Language Support:** Native Gujarati script processing.
* **Architecture:** Uses Embedding layers and GRU (Gated Recurrent Units) for sequence modeling.
* **Preprocessing:** Includes tokenization, n-gram sequence generation, and padding using TensorFlow Keras preprocessing tools.
* **Prediction:** Real-time next-word prediction function capable of handling variable input lengths.

---

## 🧠 Model Architecture
The model is built using **TensorFlow/Keras** with the following sequential structure:

1.  **Embedding Layer:** Converts tokenized words into dense vectors of fixed size.
    * *Input Dim:* ~4,352 (Vocabulary Size)
    * *Output Dim:* 4
2.  **GRU Layer:** A Recurrent Neural Network layer with 512 units to capture temporal dependencies in the text.
3.  **Dense Layer:** Output layer using the `softmax` activation function to output a probability distribution over the vocabulary.

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 39, 4)             17,408    
                                                                 
 gru (GRU)                   (None, 512)               795,648   
                                                                 
 dense (Dense)               (None, 4352)              2,232,576 
=================================================================
Total params: 3,045,632

```

## 📂 Dataset
The biggest challenge in Indic NLP is the scarcity of clean datasets. We aggregated data from:
* **Wikipedia Articles:** English articles translated into Gujarati.
* **Literature:** Gujarati storybooks and prose.

**Preprocessing Steps:**
* **Cleaning:** Removal of noise and English characters using RegEx.
* **Tokenization:** Converting text to sequences of integers.
* **Sequence Generation:** Creating N-gram sequences (e.g., "મારૂ નામ" -> "મિત") to train the model on preceding context.

---

## 📊 Results & Performance
The model was trained for **25 epochs**. As observed in the training logs:
* **Final Accuracy:** ~41.00%
* **Final Loss:** ~2.83

*Note: Research indicates that as dataset size increases (from 51KB to 3.4MB), the contextuality and accuracy of the output improve significantly.*

### Visualization
The repository includes code to generate:
* **Accuracy vs. Epoch plots**
* **Loss vs. Epoch plots**

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.x
* TensorFlow
* NumPy
* Matplotlib

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Dharmil2684/Next_word_prediction_Gujarati
   cd gujarati-next-word-predictor
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

### Running the Predictor
You can use the saved model to make predictions. Here is a snippet from the notebook:

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model("gujarati_nextword_model.keras")

def predict_next_word(seed_text):
    # ... (See repo for full preprocessing logic) ...
    predicted_word = model.predict(token_list)
    return predicted_word

# Example Usage
print(predict_next_word("આ પુસ્તકમાં ભારત અને વિદેશના ક્રિકેટ વિશે લખવામાં")) 
# Output: આવ્યું (or contextually relevant word)
```

---

## 🔮 Future Scope
To further improve the accuracy and contextuality of the text generation:
* **Bidirectional RNNs/LSTMs:** Implementing Bi-LSTMs to consider context from both future and past words, as the current unidirectional approach has limitations.
* **Larger Dataset:** Collecting a more extensive corpus of POS-tagged Gujarati text to explore supervised learning.
* **Advanced Architectures:** Experimenting with LSTMs, CNNs, or Transformers (BERT/GPT) adapted for Indic languages.

---

## 📚 References
* **Base Research:** "Next Word Predictor in Gujarati Language", *International Journal of Innovative Research in Science, Engineering and Technology (IJIRSET)*, Vol 12, Issue 10, October 2023.
* **Methodology:** Recurrent Neural Networks (RNN) & Word Embeddings.

---
*This project is for educational and research purposes.*
