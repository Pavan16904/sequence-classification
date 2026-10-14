from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    LSTM,
    GRU,
    Bidirectional,
    Dense,
)


VOCAB_SIZE = 10_000
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64


def build_model(model_name):
    """Create one sentiment classification model."""
    model_name = model_name.lower()

    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, mask_zero=True))

    if model_name == "rnn":
        model.add(SimpleRNN(HIDDEN_SIZE))

    elif model_name == "birnn":
        model.add(Bidirectional(SimpleRNN(HIDDEN_SIZE)))

    elif model_name == "deep_rnn":
        model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=True))
        model.add(SimpleRNN(HIDDEN_SIZE))

    elif model_name == "lstm":
        model.add(LSTM(HIDDEN_SIZE))

    elif model_name == "gru":
        model.add(GRU(HIDDEN_SIZE))

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model