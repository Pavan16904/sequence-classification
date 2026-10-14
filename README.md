# IMDB Sentiment Classification with Recurrent Neural Networks

A sequence classification project comparing recurrent neural network architectures on IMDB movie reviews. The models predict whether a review is positive or negative and are compared using three input sequence lengths.

## Team Members

| Roll Number | Name |
|---|---|
| 1KS22AI030 | Pavan S |
| 1KS23AI031 | Monashree P |
| 1KS23AI046 | Samana R Kashyap |
| 1KS23AI054 | Sripriya R |

## Project Overview

This project investigates how recurrent architectures classify text sequences and handle information across different sequence lengths. Five models are compared: Simple RNN, Bidirectional RNN, Deep RNN, LSTM, and GRU.

Each model is evaluated with review lengths of 64, 128, and 256 words. Test accuracy and F1 score are used to compare their performance.

## Dataset and Preprocessing

The project uses the IMDB movie review dataset provided by Keras. It contains 25,000 training reviews and 25,000 test reviews, labelled positive or negative.

Keras supplies the reviews as numerical word sequences, so a separate tokenizer is not required. The vocabulary is limited to the 10,000 most frequent words. Reviews are padded or truncated to lengths of 64, 128, and 256 tokens. The training data is split into training and validation sets; the test set is reserved for final evaluation.

## Architectures

| Architecture | Description |
|---|---|
| Simple RNN | Baseline recurrent model with one recurrent layer. |
| Bidirectional RNN | Reads each complete review in forward and reverse directions. |
| Deep RNN | Uses two stacked Simple RNN layers. |
| LSTM | Uses gates and a cell state to manage information across the sequence. |
| GRU | Uses recurrent gates with a simpler structure than LSTM. |

A Bidirectional RNN is suitable for review classification because the complete review is available at prediction time. It is not suitable for real-time next-word prediction when future words are unavailable.

## Project Requirements

| Requirement | Implementation |
|---|---|
| Collect and preprocess sequential data | Loads and prepares IMDB movie reviews. |
| Convert text to numerical sequences | Uses the word-ID sequences provided by Keras. |
| Pad and truncate sequences | Tests maximum lengths of 64, 128, and 256 words. |
| Create training, validation, and test sets | Uses a validation split from the training data and keeps the test set separate. |
| Implement recurrent architectures | Compares Simple RNN, Bidirectional RNN, Deep RNN, LSTM, and GRU. |
| Study sequence length | Evaluates each model at all three lengths. |
| Compare simple and gated RNNs | Compares Simple RNN models with LSTM and GRU. |
| Analyze gradient problems | Discusses vanishing and exploding gradients conceptually. |
| Measure performance | Reports test accuracy and F1 score. |

## Project Structure

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
| `data.py` | Loads the data, creates the validation split, and prepares sequences. |
| `models.py` | Defines the five recurrent model architectures. |
| `experiment.py` | Trains and evaluates each model at each sequence length. |
| `main.py` | Runs the experiments and saves the graphs and results. |
| `requirements.txt` | Lists the required Python packages. |
| `results/` | Contains generated graphs and the experiment results CSV. |

## Experimental Settings

| Setting | Value |
|---|---:|
| Vocabulary size | 10,000 words |
| Embedding size | 64 |
| Recurrent hidden units | 64 |
| Batch size | 64 |
| Maximum epochs | 3 |
| Early stopping | Validation loss, patience 1 |
| Sequence lengths | 64, 128, 256 |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Metrics | Accuracy and F1 score |

## Installation

Install Python, open this repository in Visual Studio Code, and open a terminal in the repository folder. Install the dependencies:

```bash
py -m pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
tensorflow
scikit-learn
matplotlib
```

## Run the Project

Run the experiment from the repository folder:

```bash
py main.py
```

Keras downloads the IMDB dataset automatically on the first run. The experiment runs each of the five architectures at all three sequence lengths. Each run trains for up to three epochs; early stopping may stop training sooner if validation loss does not improve.

## Results

| Architecture | Accuracy (64) | F1 (64) | Accuracy (128) | F1 (128) | Accuracy (256) | F1 (256) |
|---|---:|---:|---:|---:|---:|---:|
| Simple RNN | 0.698 | 0.666 | 0.791 | 0.788 | 0.663 | 0.551 |
| Bidirectional RNN | 0.767 | 0.759 | 0.773 | 0.754 | 0.712 | 0.728 |
| Deep RNN | 0.576 | 0.511 | 0.661 | 0.559 | 0.641 | 0.667 |
| LSTM | 0.775 | 0.765 | 0.811 | 0.794 | 0.851 | 0.855 |
| GRU | 0.767 | 0.773 | 0.826 | 0.821 | **0.855** | **0.856** |

The best test result was achieved by the GRU at sequence length 256, with 0.855 accuracy and an F1 score of 0.856. The LSTM at length 256 was close, with 0.851 accuracy and an F1 score of 0.855.

## Generated Files

| File | Description |
|---|---|
| `results/accuracy.png` | Training and validation accuracy curves at sequence length 128. |
| `results/loss.png` | Training and validation loss curves at sequence length 128. |
| `results/model_comparison.png` | Test accuracy by architecture and sequence length. |
| `results/model_results.csv` | Accuracy, F1 score, and training time for each experiment. |

## Discussion

The LSTM and GRU achieved their strongest results at sequence length 256 in this experiment. Longer input did not improve every architecture: the Simple RNN performed best at length 128, and the Bidirectional RNN results were similar at lengths 64 and 128. The effect of sequence length therefore depended on the model.

A vanishing gradient becomes very small as it is propagated backward through many time steps, making it difficult for a model to learn from earlier words. An exploding gradient becomes very large and can make training unstable. LSTM and GRU gates are designed to help control information flow across longer sequences. Gradient norms were not measured directly in this project.

## Conclusion

The GRU at sequence length 256 achieved the highest test accuracy and F1 score in this experiment. The LSTM at the same length performed almost as well. These results demonstrate that both architecture and sequence length can affect sentiment classification performance. Further runs with different random seeds could help assess the consistency of the results.
