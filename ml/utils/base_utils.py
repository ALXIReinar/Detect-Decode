from pathlib import Path

import torch
from matplotlib import pyplot as plt, patches

from ml.logger_config import log_event


def visualize_bboxes(image, boxes, figcolor='red', figsize=(8, 8), linewidth=1):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3:
            image = image.squeeze(0)
        image = image.numpy()

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=linewidth,
            edgecolor=figcolor,
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.show()


def check_imgs_anns_equal(dataset_location: Path):
    layouts = ['train', 'val', 'test']
    for layout in layouts:
        missing_files = []
        for file in (dataset_location / layout / "annotations").iterdir():
            if not Path.exists(dataset_location / layout / "imgs" / f"{file.name.split('.')[0]}.png"):
                missing_files.append(file)
        log_event(f"\033[34m{layout=}\033[0m | \033[31m{len(missing_files)=}\033[0m, \033[36m{missing_files=}\033[0m", level='WARNING')




def plot_curves(curves, title, xlabel="Epoch", ylabel="Value", save_path=None, show=True):
    plt.figure(figsize=(10, 6))

    for curve in curves:
        plt.plot(curve["values"], label=curve["label"], color=curve.get("color", None), linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
