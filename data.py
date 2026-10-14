from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split


VOCAB_SIZE = 10_000


def load_data(max_length=128):
    """Load IMDB reviews and prepare padded train, validation, and test data."""
    print("Downloading/loading IMDB reviews...")

    (all_train_x, all_train_y), (test_x, test_y) = imdb.load_data(
        num_words=VOCAB_SIZE
    )

    train_x, val_x, train_y, val_y = train_test_split(
        all_train_x,
        all_train_y,
        test_size=0.2,
        random_state=42,
        stratify=all_train_y,
    )

    train_x = pad_sequences(
        train_x, maxlen=max_length, padding="post", truncating="post"
    )
    val_x = pad_sequences(
        val_x, maxlen=max_length, padding="post", truncating="post"
    )
    test_x = pad_sequences(
        test_x, maxlen=max_length, padding="post", truncating="post"
    )

    return train_x, val_x, test_x, train_y, val_y, test_y