from ml.base_utils import plot_curves


def postprocess_russian(text):
    """
    Постобработка для русского текста
    Добавляет заглавные буквы и запятые
    """
    "Заглавные после точки"
    sentences = text.split('. ')
    sentences = [s.capitalize() for s in sentences]
    text = '. '.join(sentences)

    "Запятые перед союзами"
    conjunctions = ['что', 'чтобы', 'если', 'когда', 'потому', 'хотя']
    for word in conjunctions:
        text = text.replace(f' {word}', f', {word}')

    return text



def plot_lr_chronology(history_arg, save_path=None, show=False):
    """
    График Learning Rate во время обучения
    """
    curves = [
        {
            "label": "Learning Rate",
            "values": history_arg["lr"],
            "color": "#E67E22"  # Оранжевый
        }
    ]


    plot_curves(
        curves=curves,
        title="LR Chronology",
        ylabel="LR Value",
        save_path=save_path,
        show=show,
    )

def plot_loss_dynamics(history_arg, save_path=None, show=False):
    """График общих метрик валидации: total loss и mAP."""
    values_list = history_arg["general_metrics"]

    curves_total_losses = [
        {
            "label": "Total Val Loss",
            "values": values_list['val_loss_list'],
            "color": "#E74C3C"  # Красный
        },
        {
            "label": "Total Train Loss",
            "values": values_list['train_loss_list'],
            "color": "#2ECC71"  # Зеленый
        },
    ]

    plot_curves(
        curves=curves_total_losses,
        title="Validation Dynamics",
        ylabel="Loss",
        save_path=save_path,
        show=show
    )


def plot_metrics_dynamics(history_arg, save_path=None, show=False):
    """График метрик при валидации: val map@50 и map@50:95"""
    values_list = history_arg["general_metrics"]

    curves = [
        {
            "label": "CER",
            "values": values_list["cer_list"],
            "color": "#2ECC71"  # Зеленый
        },
        {
            "label": "WER",
            "values": values_list["wer_list"],
            "color": "#3498DB"  # Синий
        }
    ]

    plot_curves(
        curves=curves,
        title="Metrics Validation Dynamics",
        ylabel="CER & WER",
        save_path=save_path,
        show=show,
    )

