import csv
import os

import matplotlib.pyplot as plt

from experiment import run_experiments


RESULTS_FOLDER = "results"


def save_results(results):
    """Save scores and create three project graphs."""
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Save the numerical results in a CSV file.
    csv_path = os.path.join(RESULTS_FOLDER, "model_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "sequence_length",
                "accuracy",
                "f1_score",
                "training_seconds",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow({
                "model": result["model"],
                "sequence_length": result["sequence_length"],
                "accuracy": f'{result["accuracy"]:.4f}',
                "f1_score": f'{result["f1_score"]:.4f}',
                "training_seconds": f'{result["training_seconds"]:.1f}',
            })

    # Graph 1: training and validation accuracy, for length 128.
    selected = [r for r in results if r["sequence_length"] == 128]

    plt.figure(figsize=(9, 6))
    for result in selected:
        history = result["history"]
        epochs = range(1, len(history["accuracy"]) + 1)
        plt.plot(epochs, history["accuracy"], marker="o",
                 label=f'{result["model"]} training')
        plt.plot(epochs, history["val_accuracy"], marker=".",
                 linestyle="--", label=f'{result["model"]} validation')

    plt.title("Training and Validation Accuracy (Review Length 128)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "accuracy.png"))
    plt.close()

    # Graph 2: training and validation loss, for length 128.
    plt.figure(figsize=(9, 6))
    for result in selected:
        history = result["history"]
        epochs = range(1, len(history["loss"]) + 1)
        plt.plot(epochs, history["loss"], marker="o",
                 label=f'{result["model"]} training')
        plt.plot(epochs, history["val_loss"], marker=".",
                 linestyle="--", label=f'{result["model"]} validation')

    plt.title("Training and Validation Loss (Review Length 128)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "loss.png"))
    plt.close()

    # Graph 3: test accuracy across models and sequence lengths.
    model_names = sorted({r["model"] for r in results})
    lengths = sorted({r["sequence_length"] for r in results})
    bar_width = 0.25
    positions = list(range(len(model_names)))

    plt.figure(figsize=(10, 6))
    for index, length in enumerate(lengths):
        scores = []
        for model_name in model_names:
            match = next(
                r for r in results
                if r["model"] == model_name
                and r["sequence_length"] == length
            )
            scores.append(match["accuracy"])

        offsets = [
            position + (index - (len(lengths) - 1) / 2) * bar_width
            for position in positions
        ]
        plt.bar(offsets, scores, bar_width, label=f"Length {length}")

    plt.title("Test Accuracy by Model and Review Length")
    plt.xlabel("Model")
    plt.ylabel("Test accuracy")
    plt.xticks(positions, model_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "model_comparison.png"))
    plt.close()

    print(f"\nResults and graphs saved in the '{RESULTS_FOLDER}' folder.")


if __name__ == "__main__":
    results = run_experiments(epochs=3, batch_size=64)
    save_results(results)

    print("\nFINAL RESULTS")
    for result in results:
        print(
            f'{result["model"]:8} | '
            f'length={result["sequence_length"]:3} | '
            f'accuracy={result["accuracy"]:.3f} | '
            f'F1={result["f1_score"]:.3f}'
        )