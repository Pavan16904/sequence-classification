Replace the contents of `README.md` with this updated version. Copy everything inside the four-backtick box, but **leave out the outer four backticks**. Keep the three-backtick fences inside the README around commands and the folder diagram.

````markdown
# Sequence Classification System: IMDB Sentiment Analysis Using Recurrent Neural Networks

Deep Learning — Sequence Classification Project

## Team Members

| Roll Number | Name |
|---|---|
| 1KS22AI030 | Pavan S |
| 1KS23AI031 | Monashree P |
| 1KS23AI046 | Samana R Kashyap |
| 1KS23AI054 | Sripriya R |

## Assigned Topic

Develop a sequence classification or prediction system that receives a sequence of words, characters, sensor readings, or time-series values and predicts its class or next element. This project uses IMDB movie reviews to classify sentiment as positive or negative and compares recurrent architectures on short and long sequences.

## 1. Problem Statement and Our Approach

The goal is to classify an IMDB movie review as positive or negative. Reviews are sequences of words, so the model must learn useful information from words throughout each review.

We compare five recurrent neural network architectures at three maximum review lengths: 64, 128, and 256 words. Test accuracy and F1 score are used to study how architecture and sequence length affect classification performance.

| Project Requirement | How This Project Addresses It |
|---|---|
| Classify a sequence into a category | Predicts positive or negative sentiment for each movie review. |
| Collect and preprocess sequential data | Loads and prepares the IMDB review dataset from Keras. |
| Convert text into numerical sequences | Uses the numerical word sequences supplied by the Keras IMDB dataset. |
| Pad or truncate sequences | Evaluates reviews at lengths of 64, 128, and 256 words. |
| Compare recurrent architectures | Evaluates Simple RNN, Bidirectional RNN, Deep RNN, LSTM, and GRU. |
| Investigate sequence length | Runs each implemented architecture at all three lengths. |
| Discuss long-term dependencies and gradient problems | Explains vanishing and exploding gradients and the role of recurrent gates. |
| Measure model performance | Reports test accuracy and F1 score and saves result graphs. |

## 2. Dataset and Preprocessing

The project uses the IMDB movie review dataset provided by Keras. It contains 25,000 training reviews and 25,000 test reviews, labelled as positive or negative.

The Keras dataset already represents reviews as numerical word IDs, so a separate text tokenizer is not required. The vocabulary is limited to the 10,000 most frequent words.

The original training data is divided into training and validation sets using an 80/20 split. The separate test set is held aside for final evaluation. Reviews are padded or truncated after the final word to lengths of 64, 128, and 256 tokens.

## 3. Solution Approaches

| No. | Architecture or Technique | Project Implementation |
|---:|---|---|
| 1 | Simple RNN | Implemented as the baseline recurrent model. |
| 2 | RNN with different sequence lengths | Evaluated at lengths 64, 128, and 256. |
| 3 | Bidirectional RNN | Implemented to process each complete review in both directions. |
| 4 | Deep RNN with multiple recurrent layers | Implemented using two stacked Simple RNN layers. |
| 5 | Recursive Neural Network | Not included in the hybrid comparison; it is designed for tree-structured data. |
| 6 | LSTM | Implemented as a gated recurrent architecture. |
| 7 | GRU | Implemented as a gated recurrent architecture. |
| 8 | Hybrid comparison | Compares Simple RNN, Bidirectional RNN, Deep RNN, LSTM, and GRU. |

The Bidirectional RNN is appropriate for review classification because the complete review is available when making a prediction. It would not be suitable for real-time next-word prediction when future words are unavailable.

## 4. Project Structure

```text
sequence-classification/
├── data.py
├── models.py
├── experiment.py
├── main.py
├── requirements.txt
├── README.md
└── results/
    ├── accuracy.png
    ├── loss.png
    ├── model_comparison.png
    └── model_results.csv
```

| File or Folder | Purpose |
|---|---|
| `data.py` | Loads the IMDB dataset, creates the validation split, and pads or truncates reviews. |
| `models.py` | Defines the five recurrent model architectures. |
| `experiment.py` | Trains and evaluates each model at each sequence length. |
| `main.py` | Runs the experiments, prints results, and saves graphs and the CSV file. |
| `requirements.txt` | Lists the Python packages required to run the project. |
| `results/` | Stores the generated graphs and numerical results. |

## 5. Setup

Install Python, open this project folder in Visual Studio Code, and open **Terminal → New Terminal**.

Install the required packages from the project folder:

```bash
py -m pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
tensorflow
scikit-learn
matplotlib
```

## 6. Running the Project

Run the experiment from the project folder:

```bash
py main.py
```

On the first run, Keras downloads the IMDB dataset. The program evaluates five architectures at three sequence lengths, for up to 15 model runs. Training time depends on the computer.

Each model trains for a maximum of three epochs. Early stopping monitors validation loss and may stop a model before it reaches all three epochs.

The program saves the following files in the `results/` folder:

| Output File | Contents |
|---|---|
| `accuracy.png` | Training and validation accuracy curves for each model at sequence length 128. |
| `loss.png` | Training and validation loss curves for each model at sequence length 128. |
| `model_comparison.png` | Test accuracy comparison for all models and sequence lengths. |
| `model_results.csv` | Accuracy, F1 score, and training time for every model and sequence length. |

## 7. How the Implementation Works

### `data.py` — IMDB Data Preparation

The data loader retrieves the IMDB reviews and labels, separates a validation subset from the training data, and pads or truncates reviews to the requested sequence length. The test set remains separate for final evaluation.

### `models.py` — Recurrent Architectures

Each model begins with a 64-dimensional embedding layer and ends with a sigmoid output layer for binary classification.

| Model | Recurrent Layers |
|---|---|
| Simple RNN | One Simple RNN layer with 64 units. |
| Bidirectional RNN | One bidirectional Simple RNN layer with 64 units per direction. |
| Deep RNN | Two stacked Simple RNN layers with 64 units each. |
| LSTM | One LSTM layer with 64 units. |
| GRU | One GRU layer with 64 units. |

### `experiment.py` — Training and Evaluation

The experiment runner loops over the three sequence lengths and five model types. It trains each model using Adam and binary cross-entropy, monitors validation loss, and evaluates predictions on the test set. Accuracy, F1 score, and training time are collected.

### `main.py` — Graphs and Results

The main program runs the experiments, prints a results summary, writes `model_results.csv`, and saves three graphs in the `results/` folder.

## 8. Experimental Settings

| Setting | Value |
|---|---:|
| Vocabulary size | 10,000 words |
| Embedding size | 64 |
| Hidden units | 64 |
| Batch size | 64 |
| Maximum epochs | 3 |
| Early stopping | Validation loss, patience 1 |
| Sequence lengths | 64, 128, 256 |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Evaluation metrics | Accuracy and F1 score |

## 9. Results and Discussion

### 9.1 Test Results

The following results are from the completed experiment:

| Architecture | Accuracy (64) | F1 (64) | Accuracy (128) | F1 (128) | Accuracy (256) | F1 (256) |
|---|---:|---:|---:|---:|---:|---:|
| Simple RNN | 0.698 | 0.666 | 0.791 | 0.788 | 0.663 | 0.551 |
| Bidirectional RNN | 0.767 | 0.759 | 0.773 | 0.754 | 0.712 | 0.728 |
| Deep RNN | 0.576 | 0.511 | 0.661 | 0.559 | 0.641 | 0.667 |
| LSTM | 0.775 | 0.765 | 0.811 | 0.794 | 0.851 | 0.855 |
| GRU | 0.767 | 0.773 | 0.826 | 0.821 | **0.855** | **0.856** |

### 9.2 Effect of Sequence Length

The GRU and LSTM achieved their strongest results at length 256. The GRU at length 256 had the highest test accuracy, 0.855, and the highest F1 score, 0.856. The LSTM at the same length was close, with 0.851 accuracy and 0.855 F1.

Increasing sequence length did not improve every architecture. The Simple RNN had its best result at length 128, while the Bidirectional RNN had similar results at lengths 64 and 128. These results show that the effect of sequence length depends on the architecture.

### 9.3 Short-Term and Long-Term Dependencies

Recurrent networks process tokens in sequence and use information from earlier steps when processing later ones. A Simple RNN can have difficulty learning relationships between words that are far apart in a long review.

An LSTM uses gates and a cell state to help preserve or update information across time steps. A GRU also uses gates to control information carried through the sequence. In this experiment, the LSTM and GRU produced their strongest results with 256-word reviews.

### 9.4 Vanishing and Exploding Gradients

During training, gradients are propagated backward through the sequence. A vanishing gradient becomes very small, making it difficult for the model to learn from earlier words. An exploding gradient becomes very large and can make training unstable.

LSTM and GRU gates are designed to help control information flow across longer sequences. This project discusses these gradient problems conceptually; it does not directly record gradient norms.

## 10. Conclusion

The GRU with a sequence length of 256 achieved the best result in this experiment, with 0.855 test accuracy and an F1 score of 0.856. The LSTM at length 256 performed almost as well. The results show that both architecture and sequence length can affect sentiment classification performance. Repeating the experiments with different random seeds would help determine how consistent these results are.

## 11. Possible Extensions

- Repeat each experiment with multiple random seeds and report average scores.
- Add confidence intervals to the accuracy and F1 results.
- Record gradient norms during training to study vanishing and exploding gradients directly.
- Compare additional sequence lengths or embedding sizes.
- Add a next-word prediction task using a unidirectional recurrent model.
````
