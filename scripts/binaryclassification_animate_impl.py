

from cpe487587hw import deepl
from cpe487587hw import animation

import matplotlib.pyplot as plt
from datetime import datetime


def main():
    
    n = 40000        # number of samples
    d = 200          # number of features
    epochs = 5000
    lr = 0.01
    dt = 0.04

    
    # Train model (returns 3D weight histories)
    
    W1_list, W2_list, W3_list, W4_list, loss_history = (
        deepl.binary_classification(
            n=n,
            d=d,
            epochs=epochs,
            lr=lr
        )
    )

    
    # Animate ALL FOUR weight matrices
    
    animation.animate_weight_heatmap(
        W1_list.cpu(),
        dt=dt,
        file_name="W1_weight_evolution",
        title_str="W1 Weight Evolution"
    )

    animation.animate_weight_heatmap(
        W2_list.cpu(),
        dt=dt,
        file_name="W2_weight_evolution",
        title_str="W2 Weight Evolution"
    )

    animation.animate_weight_heatmap(
        W3_list.cpu(),
        dt=dt,
        file_name="W3_weight_evolution",
        title_str="W3 Weight Evolution"
    )

    animation.animate_weight_heatmap(
        W4_list.cpu(),
        dt=dt,
        file_name="W4_weight_evolution",
        title_str="W4 Weight Evolution"
    )

    
    # Loss vs Epoch plot
    
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross-Entropy Loss vs Epochs")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plt.savefig(f"crossentropyloss_{timestamp}.pdf")
    plt.close()

    print("ALL FOUR weight animations created successfully")


if __name__ == "__main__":
    main()
