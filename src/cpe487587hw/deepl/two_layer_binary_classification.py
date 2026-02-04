import torch
def binary_classification(n,d,epochs=10000,lr=0.001):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Generate data
    X = torch.randn(n, d, dtype=torch.float32, device=device)
    Y=(X.sum(dim=1,keepdim=True) > 2).float().to(device)
    W1=torch.randn(d,48,device=device)*(2/d)**0.5
    W1.requires_grad_()

    
    W2=torch.randn(48,16,device=device)*(2/48)**0.5
    W2.requires_grad_()
    W3=torch.randn(16,32,device=device)*(2/16)**0.5
    W3.requires_grad_()
    W4=torch.randn(32,1,device=device)*(2/32)**0.5
    W4.requires_grad_()
    loss_history=[]
    for epoch in range(epochs):
        Z1=X.mm(W1)
        A1=1/(1+torch.exp(-(Z1.mm(W2))))
        Z2=A1.mm(W3)
        A2=1/(1+torch.exp(-(Z2.mm(W4))))
        loss=-(Y*torch.log(A2)+(1-Y)*torch.log(1-A2)).sum()
        loss_history.append(loss.item())
        loss.backward()
        with torch.no_grad():
            W4-=lr*W4.grad
            W3-=lr*W3.grad
            W2-=lr*W2.grad
            W1-=lr*W1.grad
            W4.grad.zero_()
            W3.grad.zero_()
            W2.grad.zero_()
            W1.grad.zero_()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
    return W1, W2, W3, W4, loss_history

