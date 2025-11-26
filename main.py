import argparse
import torch.nn as nn
import torch.optim as optim

# Assuming models and utils are importable
from nets import LeNet5, AlexNet, VGG16
from utils.data_loader import get_dataloaders
from utils.train_eval_save import train_and_save_model

# Dictionary mapping model name strings to their class constructors
MODEL_FACTORY = {
    "lenet5": LeNet5,
    "alexnet": AlexNet,
    "vgg16": VGG16,
}


def main():
    """
    Parses command-line arguments to select a model and set hyperparameters,
    then initializes and trains the model.
    """
    parser = argparse.ArgumentParser(
        description="Run a deep learning model training experiment with configurable hyperparameters."
    )

    # --- Positional Argument (Required) ---
    parser.add_argument(
        "model_name",
        type=str,
        choices=MODEL_FACTORY.keys(),
        help="The name of the model to train (must be one of: lenet5, alexnet, vgg16).",
    )

    # --- Optional Hyperparameter Arguments ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs. (Default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for the SGD optimizer. (Default: 0.01)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data loaders. (Default: 64)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer. (Default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty) for SGD optimizer. (Default: 1e-4)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of output classes for the final layer. (Default: 10)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="Dataset name to use (e.g., MNIST). (Default: MNIST)",
    )

    args = parser.parse_args()

    # 1. Model Selection and Instantiation
    ModelClass = MODEL_FACTORY[args.model_name]
    print(f"--- Initializing Model: {args.model_name.upper()} ---")
    model = ModelClass(num_classes=args.num_classes)

    # 2. Setup Optimizer with Command-Line Args
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 3. Setup Loss Function and Data Loaders
    loss_fn = nn.CrossEntropyLoss()

    data_loader = get_dataloaders(
        dataset_name=args.dataset, batch_size=args.batch_size
    )

    # 4. Define Save Path
    save_path = f".models/{args.model_name}.pth"
    print(
        f"Training for {args.epochs} epochs with LR={args.lr}, Batch Size={args.batch_size}"
    )

    # 5. Run Training
    train_and_save_model(
        model=model,
        epochs=args.epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_loader=data_loader,
        save_path=save_path,
    )

    print(f"--- Training Complete. Model saved to {save_path} ---")


if __name__ == "__main__":
    main()
