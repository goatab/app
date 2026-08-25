from pathlib import Path

import joblib
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 128


def _hog_features(gray_image):
    pixels = np.asarray(gray_image, dtype=np.float32) / 255.0
    vertical = np.diff(pixels, axis=0, append=pixels[-1:, :])
    horizontal = np.diff(pixels, axis=1, append=pixels[:, -1:])
    magnitudes = np.sqrt(horizontal**2 + vertical**2)
    angles = (np.arctan2(vertical, horizontal) * 180 / np.pi) % 180

    cell_size = 8
    cells_per_side = IMAGE_SIZE // cell_size
    histograms = []
    for row in range(cells_per_side):
        for column in range(cells_per_side):
            row_slice = slice(row * cell_size, (row + 1) * cell_size)
            column_slice = slice(column * cell_size, (column + 1) * cell_size)
            cell_angles = angles[row_slice, column_slice].ravel()
            cell_magnitudes = magnitudes[row_slice, column_slice].ravel()
            histogram, _ = np.histogram(
                cell_angles,
                bins=9,
                range=(0, 180),
                weights=cell_magnitudes,
            )
            histograms.append(histogram)

    return np.asarray(histograms, dtype=np.float32).ravel()


def image_features(image_path):
    with Image.open(image_path) as image:
        color_image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        gray_image = color_image.convert("L")
        pixels = np.asarray(color_image, dtype=np.float32) / 255.0

    color_histograms = [
        np.histogram(pixels[:, :, channel], bins=16, range=(0, 1), density=True)[0]
        for channel in range(3)
    ]
    resized_pixels = np.asarray(gray_image, dtype=np.float32).ravel() / 255.0
    return np.concatenate(
        [resized_pixels, _hog_features(gray_image), *color_histograms]
    ).astype(np.float32)


def load_image_folder(dataset_dir):
    dataset_path = Path(dataset_dir)
    features = []
    labels = []

    for label_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        image_paths = sorted(
            path for path in label_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            try:
                features.append(image_features(image_path))
                labels.append(label_dir.name)
            except (OSError, ValueError) as error:
                print(f"Skipping {image_path}: {error}")

    if not features:
        raise ValueError(
            f"No readable images found in {dataset_path}. "
            "Use one folder per class, for example data/images/burn/*.jpg."
        )

    return np.asarray(features), np.asarray(labels)


def train_classifier(dataset_dir, model_path="models/injury_classifier.joblib"):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    features, labels = load_image_folder(dataset_dir)
    class_names, class_counts = np.unique(labels, return_counts=True)
    if len(class_names) < 2:
        raise ValueError("Training requires at least two class folders.")
    if np.min(class_counts) < 2:
        raise ValueError("Each class needs at least two readable images.")

    classifier = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True),
    )
    classifier.fit(features, labels)

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_path)
    return classifier, len(features), class_names.tolist()


def predict_image(image_path, model_path="models/injury_classifier.joblib"):
    classifier = joblib.load(model_path)
    probabilities = classifier.predict_proba([image_features(image_path)])[0]
    best_index = int(np.argmax(probabilities))
    return {
        "label": str(classifier.classes_[best_index]),
        "confidence": float(probabilities[best_index]),
    }