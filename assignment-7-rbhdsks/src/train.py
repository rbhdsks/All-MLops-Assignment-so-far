import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18


def build_model(num_classes=10):
    # Pre-trained ResNet18, adapted for MNIST (1 channel, 10 classes)
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_loaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    # Small subset to keep it fast — remove the Subset wrapping if you want full MNIST
    train_ds = torch.utils.data.Subset(train_ds, range(5000))
    test_ds = torch.utils.data.Subset(test_ds, range(1000))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def save_sample_predictions(model, loader, device, path):
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].cpu().squeeze(), cmap="gray")
        ax.set_title(f"pred={pred[i].item()}, true={y[i].item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)
    train_loader, test_loader = get_loaders(args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    with mlflow.start_run():
        # ---- Log params ----
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("architecture", "resnet18_mnist")
        mlflow.log_param("optimizer", "SGD_momentum0.9")

        # ---- Train ----
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
            train_loss = running_loss / len(train_loader.dataset)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)

            # ---- Log metrics per epoch ----
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # ---- Sample prediction plot ----
        os.makedirs("artifacts", exist_ok=True)
        plot_path = "artifacts/sample_predictions.png"
        save_sample_predictions(model, test_loader, device, plot_path)
        mlflow.log_artifact(plot_path)

        # ---- Save model weights as artifact (visible in Artifacts tab) ----
        model_path = "artifacts/model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # ---- Log the model with signature + input example (for serving) ----
        sample_input = np.random.rand(1, 1, 28, 28).astype(np.float32)
        model.eval()
        with torch.no_grad():
            sample_output = model(torch.from_numpy(sample_input).to(device)).cpu().numpy()
        signature = infer_signature(sample_input, sample_output)

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
        )
        print("Run complete. Model + artifacts logged to MLflow.")


if __name__ == "__main__":
    main()