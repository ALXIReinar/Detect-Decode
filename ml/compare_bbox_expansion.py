"""
Сравнение baseline vs bbox expansion для оценки улучшения.
"""
from pathlib import Path

from ml.base_utils import compare_pipeline_scores
from ml.config import WORKDIR
from ml.test_ocr_pipeline_stage import test_run
from ml.logger_config import log_event


def compare_bbox_expansion():
    """
    Сравнивает результаты с и без bbox expansion.
    """
    detector_weights = WORKDIR / 'ml' / 'detector' / 'model_weights' / 'best.pt'
    word_decoder_weights = WORKDIR / 'ml' / 'word_decoder' / 'model_weights' / 'best.pt'
    
    # Baseline (без bbox expansion)
    log_event("\n\033[35m=== BASELINE (без bbox expansion) ===\033[0m", level='WARNING')
    baseline = {
        'pipeline_score': 77.64,
        'cer': 22.44,
        'wer': 40.74,
        'map50': 94.70,
        'map50_95': 88.26
    }
    log_event(
        f"Pipeline Score: {baseline['pipeline_score']:.2f}%\n"
        f"CER: {baseline['cer']:.2f}% | WER: {baseline['wer']:.2f}%\n"
        f"mAP@50: {baseline['map50']:.2f}% | mAP@50-95: {baseline['map50_95']:.2f}%",
        level='INFO'
    )
    
    # Improved (с bbox expansion 5%)
    log_event("\n\033[35m=== IMPROVED (bbox expansion 5%) ===\033[0m", level='WARNING')
    improved = test_run(
        detector_weights_path=detector_weights,
        word_decoder_weights_path=word_decoder_weights,
        workers=4,
        batch_size=8
    )
    
    # Сравнение
    log_event("\n\033[35m=== COMPARISON ===\033[0m", level='WARNING')
    diff = compare_pipeline_scores(baseline, improved)
    
    log_event(
        f"Pipeline Score: {diff['pipeline_delta']:+.2f}% ({baseline['pipeline_score']:.2f}% → {improved['pipeline_score']:.2f}%)\n"
        f"CER: {diff['cer_delta']:+.2f}% ({baseline['cer']:.2f}% → {improved['cer']:.2f}%)\n"
        f"WER: {diff['wer_delta']:+.2f}% ({baseline['wer']:.2f}% → {improved['wer']:.2f}%)\n"
        f"Detector: {diff['detector_delta']:+.2f}% ({baseline['map50_95']:.2f}% → {improved['map50_95']:.2f}%)\n"
        f"Decoder: {diff['decoder_delta']:+.2f}%\n"
        f"Is improvement: \033[{'32' if diff['is_improvement'] else '31'}m{diff['is_improvement']}\033[0m",
        level='WARNING'
    )
    
    return {
        'baseline': baseline,
        'improved': improved,
        'diff': diff
    }


if __name__ == '__main__':
    results = compare_bbox_expansion()
    
    if results['diff']['is_improvement']:
        print(f"\n✅ Bbox expansion улучшил метрики на {results['diff']['pipeline_delta']:+.2f}%!")
    else:
        print(f"\n❌ Bbox expansion ухудшил метрики на {results['diff']['pipeline_delta']:+.2f}%")
