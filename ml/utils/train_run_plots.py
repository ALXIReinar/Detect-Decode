"""
Функции для графиков по лоссам и метрикам
"""

from ml.utils.base_utils import plot_curves


def plot_validation_metrics(history_arg, save_path=None, show=False):
    """График общих метрик валидации: total loss и mAP."""
    values_list = history_arg["general_metrics"]
    *_, total_val_losses = zip(*values_list["val_loss_list"])

    curves = [
        {
            "label": "Total Val Loss",
            "values": total_val_losses,
            "color": "#E74C3C"  # Красный
        },
        {
            "label": "mAP@0.5",
            "values": values_list["map50_list"],
            "color": "#3498DB"  # Синий
        },
        {
            "label": "mAP@0.5:0.95",
            "values": values_list["map5095_list"],
            "color": "#2ECC71"  # Зеленый
        }
    ]

    plot_curves(
        curves=curves,
        title="Validation Metrics",
        ylabel="Loss / mAP",
        save_path=save_path,
        show=show
    )


def plot_training_dynamics(history_arg, save_path=None, show=False):
    """График динамики обучения: train/val loss и learning rate."""
    values_list = history_arg["general_metrics"]
    *_, total_train_losses = zip(*values_list["train_loss_list"])
    *_, total_val_losses = zip(*values_list["val_loss_list"])

    curves = [
        {
            "label": "Train Loss",
            "values": total_train_losses,
            "color": "#E67E22"  # Оранжевый
        },
        {
            "label": "Val Loss",
            "values": total_val_losses,
            "color": "#E74C3C"  # Красный
        },
        {
            "label": "Learning Rate",
            "values": history_arg["lr"],
            "color": "#9B59B6"  # Фиолетовый
        }
    ]

    plot_curves(
        curves=curves,
        title="Training Dynamics",
        ylabel="Loss / LR",
        save_path=save_path,
        show=show,
    )


def plot_train_val_box_cls_dfl(history_arg, save_path=None, show=False):
    """
    График компонентов loss: box, cls, dfl для train и val.
    """
    values_list = history_arg["general_metrics"]

    val_box_losses, val_cls_losses, val_dfl_losses, _ = zip(*values_list["val_loss_list"])
    train_box_losses, train_cls_losses, train_dfl_losses, _ = zip(*values_list["train_loss_list"])

    curves = [
        # Validation losses
        {
            "label": "Val Box Loss",
            "values": val_box_losses,
            "color": "#E74C3C"  # Красный
        },
        {
            "label": "Val Cls Loss",
            "values": val_cls_losses,
            "color": "#3498DB"  # Синий
        },
        {
            "label": "Val DFL Loss",
            "values": val_dfl_losses,
            "color": "#2ECC71"  # Зеленый
        },
        {
            "label": "Train Box Loss",
            "values": train_box_losses,
            "color": "#C0392B"  # Темно-красный
        },
        {
            "label": "Train Cls Loss",
            "values": train_cls_losses,
            "color": "#2980B9"  # Темно-синий
        },
        {
            "label": "Train DFL Loss",
            "values": train_dfl_losses,
            "color": "#27AE60"  # Темно-зеленый
        }
    ]

    plot_curves(
        curves=curves,
        title="Loss Components: Box, Cls, DFL",
        ylabel="Loss Value",
        save_path=save_path,
        show=show,
    )

