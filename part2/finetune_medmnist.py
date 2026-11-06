#!/usr/bin/env python3
"""Fine-tune an ImageNet ResNet-18 on the PneumoniaMNIST dataset.

The default configuration freezes the convolutional backbone and trains the
classification head.  Use --unfreeze-layer4 for a deeper (and slower) fine-tune.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def require_training_packages():
    try:
        import medmnist
        import torch
        import torch.nn as nn
        from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Part 2 requires torch, torchvision, medmnist, and scikit-learn. "
            "Install hw4/requirements.txt or run the Colab notebook."
        ) from exc
    return medmnist, torch, nn, confusion_matrix, roc_auc_score, roc_curve, DataLoader, models, transforms


def seed_everything(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def build_datasets(image_size: int, medmnist, transforms):
    info = medmnist.INFO["pneumoniamnist"]
    data_class = getattr(medmnist, info["python_class"])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_data = data_class(split="train", transform=train_transform, download=True, size=28)
    val_data = data_class(split="val", transform=eval_transform, download=True, size=28)
    test_data = data_class(split="test", transform=eval_transform, download=True, size=28)
    return train_data, val_data, test_data, info


def build_model(models, nn, unfreeze_layer4: bool):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    if unfreeze_layer4:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    return model


def run_epoch(model, loader, criterion, optimizer, device, torch_module) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    count = 0
    context = torch_module.enable_grad() if training else torch_module.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze(1).long().to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += int(labels.size(0))
    return loss_sum / max(count, 1), correct / max(count, 1)


def collect_predictions(model, loader, device, torch_module) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []
    with torch_module.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch_module.softmax(logits, dim=1)[:, 1]
            labels_all.append(labels.squeeze(1).numpy())
            probabilities_all.append(probabilities.cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(probabilities_all)


def evaluate_binary(labels, probabilities, confusion_matrix, roc_auc_score) -> Dict[str, object]:
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    accuracy = (tp + tn) / max(matrix.sum(), 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }


def save_plots(history, labels, probabilities, metrics, roc_curve, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="train")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1), title="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "part2_training_history.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray(metrics["confusion_matrix"])
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            axes[0].text(col, row, str(matrix[row, col]), ha="center", va="center")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["normal", "pneumonia"], yticklabels=["normal", "pneumonia"], xlabel="Predicted", ylabel="True", title="Test confusion matrix")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].plot(false_positive_rate, true_positive_rate, color="#165D8C", label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="0.55")
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Test ROC", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "part2_test_evaluation.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-layer4", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("part2/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packages = require_training_packages()
    medmnist, torch, nn, confusion_matrix, roc_auc_score, roc_curve, DataLoader, models, transforms = packages
    seed_everything(args.seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device={device}")

    train_data, val_data, test_data, info = build_datasets(args.image_size, medmnist, transforms)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(models, nn, args.unfreeze_layer4).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_state = None
    best_val_accuracy = -1.0
    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, torch)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, device, torch)
        for key, value in (("train_loss", train_loss), ("train_accuracy", train_accuracy), ("val_loss", val_loss), ("val_accuracy", val_accuracy)):
            history[key].append(value)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        print(f"epoch {epoch + 1:02d}: train_acc={train_accuracy:.3f} val_acc={val_accuracy:.3f}")

    if best_state is None:
        raise RuntimeError("No training epoch completed.")
    model.load_state_dict(best_state)
    labels, probabilities = collect_predictions(model, test_loader, device, torch)
    metrics = evaluate_binary(labels, probabilities, confusion_matrix, roc_auc_score)
    metrics.update(
        {
            "model": "ImageNet-pretrained ResNet-18",
            "dataset": "PneumoniaMNIST",
            "task": info["task"],
            "best_validation_accuracy": float(best_val_accuracy),
            "epochs": args.epochs,
            "image_size": args.image_size,
            "unfreeze_layer4": bool(args.unfreeze_layer4),
            "device": str(device),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_plots(history, labels, probabilities, metrics, roc_curve, args.output_dir)
    (args.output_dir / "part2_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(best_state, args.output_dir / "resnet18_pneumoniamnist.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

