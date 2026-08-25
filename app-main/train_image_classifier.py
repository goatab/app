import argparse

from application.image_classifier import train_classifier


def main():
    parser = argparse.ArgumentParser(description="Train a classical image classifier.")
    parser.add_argument("dataset", help="Folder containing one labeled folder per class")
    parser.add_argument(
        "--model",
        default="models/injury_classifier.joblib",
        help="Output path for the trained model",
    )
    arguments = parser.parse_args()

    _, image_count, class_names = train_classifier(arguments.dataset, arguments.model)
    print(f"Trained on {image_count} images across: {', '.join(class_names)}")
    print(f"Saved model to {arguments.model}")


if __name__ == "__main__":
    main()