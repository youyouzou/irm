import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x,y,_ in loader:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            pred = pred.argmax(1)

            correct += (pred == y).sum().item()

            total += y.size(0)

    return correct / total