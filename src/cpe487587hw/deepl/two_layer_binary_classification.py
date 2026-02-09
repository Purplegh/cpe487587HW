import torch

def binary_classification(n, d, epochs=10000, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate data
    X = torch.randn(n, d, dtype=torch.float32, device=device)
    Y = (X.sum(dim=1, keepdim=True) > 2).float().to(device)

    # Initialize weights
    W1 = torch.randn(d, 48, device=device) * (2/d)**0.5
    W2 = torch.randn(48, 16, device=device) * (2/48)**0.5
    W3 = torch.randn(16, 32, device=device) * (2/16)**0.5
    W4 = torch.randn(32, 1, device=device) * (2/32)**0.5

    W1.requires_grad_()
    W2.requires_grad_()
    W3.requires_grad_()
    W4.requires_grad_()

  
    W1_list = torch.zeros(epochs, *W1.shape, device=device)
    W2_list = torch.zeros(epochs, *W2.shape, device=device)
    W3_list = torch.zeros(epochs, *W3.shape, device=device)
    W4_list = torch.zeros(epochs, *W4.shape, device=device)

    loss_history = []

    for epoch in range(epochs):
        # Forward pass
        Z1 = X.mm(W1)
        A1 = torch.sigmoid(Z1.mm(W2))
        Z2 = A1.mm(W3)
        A2 = torch.sigmoid(Z2.mm(W4))

        loss = -(Y * torch.log(A2) + (1 - Y) * torch.log(1 - A2)).sum()
        loss_history.append(loss.item())

        # Backward pass
        loss.backward()

        with torch.no_grad():
            W4 -= lr * W4.grad
            W3 -= lr * W3.grad
            W2 -= lr * W2.grad
            W1 -= lr * W1.grad

            W4.grad.zero_()
            W3.grad.zero_()
            W2.grad.zero_()
            W1.grad.zero_()

    
        W1_list[epoch] = W1.clone()
        W2_list[epoch] = W2.clone()
        W3_list[epoch] = W3.clone()
        W4_list[epoch] = W4.clone()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return W1_list, W2_list, W3_list, W4_list, loss_history
