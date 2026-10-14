## Results and Conclusion

We compared five recurrent neural network models on the IMDB movie review sentiment dataset. Each model was tested with review lengths of 64, 128, and 256 words. We measured test accuracy and F1 score.

The best result was achieved by the GRU model with a sequence length of 256 words. It reached 85.5% test accuracy and an F1 score of 0.856. The LSTM model at length 256 performed almost as well, with 85.1% accuracy and an F1 score of 0.855.

The LSTM and GRU models improved as sequence length increased in our experiments. The Simple RNN and Deep RNN results varied across lengths. The Deep RNN had the lowest accuracy at lengths 64 and 128.

These results suggest that gated recurrent models, especially GRU and LSTM, performed well for this sentiment classification task. We trained each model for up to three epochs, so results may vary across repeated runs.