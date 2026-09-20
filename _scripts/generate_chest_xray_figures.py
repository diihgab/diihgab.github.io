from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "assets" / "img"
SOURCE_XRAY = IMAGE_DIR / "raio-x-torax-sintetico.jpg"

CLASS_COUNTS = {
    "Atelectasia": 106,
    "Cardiomegalia": 20,
    "Consolidação": 33,
    "Edema": 16,
    "Derrame pleural": 128,
    "Enfisema": 13,
    "Fibrose": 14,
    "Hérnia": 2,
    "Infiltração": 175,
    "Massa": 45,
    "Nódulo": 54,
    "Espessamento pleural": 21,
    "Pneumonia": 10,
    "Pneumotórax": 38,
}

# The reference reports numeric values for epochs 1, 2, and 25.
REPORTED_EPOCHS = np.array([1, 2, 25])
REPORTED_LOSS = np.array([0.5352, 0.4899, 0.45])
REPORTED_ACCURACY = np.array([0.522, 0.562, 0.53])

# Transcribed from the ROC-AUC chart in the reference experiment.
ROC_AUC_HISTORY = np.array([
    0.4867, 0.5037, 0.4980, 0.5470, 0.5170,
    0.5630, 0.5560, 0.5340, 0.5860, 0.5770,
    0.5790, 0.6160, 0.6090, 0.6200, 0.6170,
    0.6060, 0.6270, 0.6330, 0.6370, 0.6440,
    0.6170, 0.5950, 0.6530, 0.6490, 0.6640,
])


def save_figure(filename: str) -> None:
    plt.savefig(IMAGE_DIR / filename, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


def raw_xray_figure() -> None:
    image = Image.open(SOURCE_XRAY).convert("L")
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    plt.figure(figsize=(10, 8))
    plot = plt.imshow(pixels, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("Raw Chest X Ray Image", fontsize=24)
    plt.colorbar(plot)
    plt.tight_layout()
    save_figure("raw-chest-x-ray-image.png")


def transformed_xray_figure() -> None:
    image = Image.open(SOURCE_XRAY).convert("L")
    image = image.resize((320, 320), Image.Resampling.BILINEAR)
    image = image.rotate(7, resample=Image.Resampling.BILINEAR)

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (pixels - 0.485) / 0.229

    plt.figure(figsize=(10, 8))
    plot = plt.imshow(normalized, cmap="gray", vmin=-2.1, vmax=2.3)
    plt.title("Transformed Chest X Ray Image", fontsize=24)
    plt.colorbar(plot)
    plt.tight_layout()
    save_figure("transformed-chest-x-ray-image.png")


def class_distribution_figure() -> None:
    labels = list(CLASS_COUNTS.keys())
    values = list(CLASS_COUNTS.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, axis = plt.subplots(figsize=(12, 8))
    bars = axis.barh(labels, values, color=colors)
    axis.invert_yaxis()
    axis.set_title("Distribuição das classes no conjunto de treino", fontsize=20)
    axis.set_xlabel("Número de imagens positivas", fontsize=14)
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.bar_label(bars, padding=5, fontsize=10)
    fig.tight_layout()
    save_figure("distribuicao-classes-raio-x.png")


def reported_metric_figure(values: np.ndarray, ylabel: str, title: str, filename: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(REPORTED_EPOCHS, values, marker="o", linewidth=2.2, color="#287bb5")
    axis.set_xticks(REPORTED_EPOCHS)
    axis.set_xlabel("Época de treinamento", fontsize=14)
    axis.set_ylabel(ylabel, fontsize=14)
    axis.set_title(title, fontsize=18)
    axis.grid(alpha=0.2)

    for epoch, value in zip(REPORTED_EPOCHS, values):
        axis.annotate(f"{value:.4f}", (epoch, value), xytext=(0, 9),
                      textcoords="offset points", ha="center", fontsize=10)

    fig.tight_layout()
    save_figure(filename)


def roc_auc_figure() -> None:
    epochs = np.arange(1, len(ROC_AUC_HISTORY) + 1)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(epochs, ROC_AUC_HISTORY, linewidth=2.2, color="#287bb5")
    axis.set_xlabel("Época de treinamento", fontsize=14)
    axis.set_ylabel("ROC-AUC médio", fontsize=14)
    axis.set_title("ROC-AUC ao longo das épocas", fontsize=18)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    save_figure("roc-auc-epocas.png")


def final_model_comparison_figure() -> None:
    metrics = ["Loss", "Acurácia", "ROC-AUC"]
    cnn = [0.45, 0.53, 0.66]
    resnet = [0.12, 0.77, 0.98]
    x = np.arange(len(metrics))
    width = 0.34

    fig, axis = plt.subplots(figsize=(10, 6))
    cnn_bars = axis.bar(x - width / 2, cnn, width, label="CNN", color="#5aa6c8")
    resnet_bars = axis.bar(x + width / 2, resnet, width, label="ResNet18", color="#e08b68")
    axis.set_xticks(x, metrics)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Valor reportado", fontsize=14)
    axis.set_title("Métricas finais reportadas por modelo", fontsize=18)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.bar_label(cnn_bars, fmt="%.2f", padding=4)
    axis.bar_label(resnet_bars, fmt="%.2f", padding=4)
    fig.tight_layout()
    save_figure("comparacao-final-cnn-resnet18.png")


def main() -> None:
    raw_xray_figure()
    transformed_xray_figure()
    class_distribution_figure()
    reported_metric_figure(
        REPORTED_LOSS,
        "Loss",
        "Loss nas épocas com valores reportados",
        "loss-epocas-cnn.png",
    )
    reported_metric_figure(
        REPORTED_ACCURACY,
        "Acurácia",
        "Acurácia nas épocas com valores reportados",
        "acuracia-epocas-cnn.png",
    )
    roc_auc_figure()
    final_model_comparison_figure()


if __name__ == "__main__":
    main()
