from cpe487587hw import deepl
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    # Run training
    W1, W2, W3, W4, loss_history = deepl. binary_classification(
        n=100,
        d=5,
        epochs=2000,
        lr=0.001
    )

    # Create loss vs epoch plot
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross-Entropy Loss vs Epochs")

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"crossentropyloss_{timestamp}.pdf"

    # Save plot
    plt.savefig(filename)
    plt.close()

    print(f"Saved plot as {filename}")

if __name__ == "__main__":
    main()

