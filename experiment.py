import time

from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

from data import load_data
from models import build_model


MODEL_NAMES = ["rnn", "birnn", "deep_rnn", "lstm", "gru"]
SEQUENCE_LENGTHS = [64, 128, 256]


def run_experiments(epochs=3, batch_size=64):
    results = []

    for max_length in SEQUENCE_LENGTHS:
        print(f"\nPreparing reviews with sequence length {max_length}...")
        train_x, val_x, test_x, train_y, val_y, test_y = load_data(max_length)

        for model_name in MODEL_NAMES:
            print(f"\nTraining {model_name.upper()} at length {max_length}...")
            model = build_model(model_name)

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=1,
                restore_best_weights=True,
            )

            start_time = time.time()

            history = model.fit(
                train_x,
                train_y,
                validation_data=(val_x, val_y),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1,
            )

            training_seconds = time.time() - start_time

            probabilities = model.predict(test_x, verbose=0).ravel()
            predictions = (probabilities >= 0.5).astype(int)

            accuracy = accuracy_score(test_y, predictions)
            f1 = f1_score(test_y, predictions)

            results.append({
                "model": model_name.upper(),
                "sequence_length": max_length,
                "accuracy": accuracy,
                "f1_score": f1,
                "training_seconds": training_seconds,
                "history": history.history,
            })

            print(
                f"Test accuracy: {accuracy:.3f} | "
                f"F1 score: {f1:.3f} | "
                f"Training time: {training_seconds:.1f} seconds"
            )

    return results