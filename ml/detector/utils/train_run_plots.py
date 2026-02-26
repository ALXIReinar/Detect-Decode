"""
Функции для графиков по лоссам и метрикам
"""

from ml.base_utils import plot_curves


def plot_loss_dynamics(history_arg, save_path=None, show=False):
    """График общих метрик валидации: total loss и mAP."""
    values_list = history_arg["general_metrics"]
    val_box_losses, val_cls_losses, val_dfl_losses, total_val_losses = zip(*values_list["val_loss_list"])
    train_box_losses, train_cls_losses, train_dfl_losses, total_train_losses = zip(*values_list["train_loss_list"])


    curves_total_losses = [
        {
            "label": "Total Val Loss",
            "values": total_val_losses,
            "color": "#E74C3C"  # Красный
        },
        {
            "label": "Total Train Loss",
            "values": total_train_losses,
            "color": "#2ECC71"  # Зеленый
        },
    ]
    curves_box_losses = [
        {
            "label": "Box Val Loss",
            "values": val_box_losses,
            "color": "#3498DB"  # Синий
        },
        {
            "label": "Box Train Loss",
            "values": train_box_losses,
            "color": "#E74C3C"  # Красный
        },
    ]
    curves_cls_losses = [
        {
            "label": "Cls Val Loss",
            "values": val_cls_losses,
            "color": "#9B59B6"  # Фиолетовый
        },
        {
            "label": "Cls Train Loss",
            "values": train_cls_losses,
            "color": "#E67E22"  # Оранжевый
        },
    ]
    curves_dfl_losses = [
        {
            "label": "DFL Val Loss",
            "values": val_cls_losses,
            "color": "#3498DB"  # Синий
        },
        {
            "label": "DFL Train Loss",
            "values": train_cls_losses,
            "color": "#2ECC71"  # Зелёный
        },
    ]


    plot_curves(
        curves=curves_total_losses,
        title="Validation Dynamics",
        ylabel="Loss",
        save_path=save_path,
        show=show
    )
    plot_curves(
        curves=curves_box_losses,
        title="Box Loss Dynamics",
        ylabel="Loss",
        save_path=save_path,
        show=show
    )
    plot_curves(
        curves=curves_cls_losses,
        title="Cls Loss Dynamics",
        ylabel="Loss",
        save_path=save_path,
        show=show
    )
    plot_curves(
        curves=curves_dfl_losses,
        title="DFL Loss Dynamics",
        ylabel="Loss",
        save_path=save_path,
        show=show
    )


def plot_metrics_dynamics(history_arg, save_path=None, show=False):
    """График метрик при валидации: val map@50 и map@50:95"""
    values_list = history_arg["general_metrics"]

    curves = [
        {
            "label": "mAP@0.5",
            "values": values_list["map50_list"],
            "color": "#2ECC71"  # Зеленый
        },
        {
            "label": "mAP@0.5:0.95",
            "values": values_list["map5095_list"],
            "color": "#3498DB"  # Синий
        }
    ]

    plot_curves(
        curves=curves,
        title="Metrics Validation Dynamics",
        ylabel="mAP@0.5 mAP@0.5:0.95",
        save_path=save_path,
        show=show,
    )

