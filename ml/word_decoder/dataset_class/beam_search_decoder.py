from pathlib import Path
from typing import Optional

import torch
from torchaudio.models.decoder import ctc_decoder

# Пытаемся импортировать CUDA decoder (может быть недоступен в зависимости от версии torchaudio)
try:
    from torchaudio.models.decoder import cuda_ctc_decoder
    CUDA_DECODER_AVAILABLE = True
except ImportError:
    CUDA_DECODER_AVAILABLE = False


class BeamSearchDecoder:
    """
    Wrapper для TorchAudio CTC beam search decoder с поддержкой CUDA(Нестабилен, баги с управлением памяти GPU).
    
    Автоматически выбирает между CUDA и CPU версией в зависимости от:
    1. Доступности cuda_ctc_decoder
    2. Наличия CUDA на устройстве
    3. Параметра use_cuda
    
    CUDA decoder (cuda_ctc_decoder):
    - Работает на CUDA тензорах
    - Упрощённый API (меньше параметров)
    - Рекомендуемые параметры: beam_size=10, blank_skip_threshold=0.95
    - Даёт 45x ускорение на батчах
    
    CPU decoder (ctc_decoder):
    - Работает только на CPU тензорах
    - Поддерживает lexicon и language model
    - Больше параметров для настройки

    """
    
    def __init__(
        self,
        tokens: list[str],
        lm_path: Optional[Path | str] = None,
        lexicon_path: Optional[Path | str] = None,
        use_cuda: bool = True,
        beam_size: int = 10,
        beam_size_token: Optional[int] = None,
        beam_threshold: float = 50.0,
        lm_weight: float = 2.0,
        word_score: float = 0.0,
        blank_token: str = "<blank>",
        sil_token: str = "<blank>",
        unk_word: str = "<unk>",
        nbest: int = 1,
        blank_skip_threshold: float = 0.95
    ):
        """
        Инициализация beam search decoder.
        
        Args:
            tokens: список токенов (charset) в том же порядке, что и в модели
            lm_path: путь к KenLM language model (только для CPU decoder)
            lexicon_path: путь к lexicon файлу (только для CPU decoder)
            use_cuda: использовать CUDA decoder если доступен (default: True)
            beam_size: размер beam для декодирования (default: 10)
            beam_size_token: размер beam на уровне токенов (только для CPU, default: None = beam_size)
            beam_threshold: порог для pruning (только для CPU, default: 50.0)
            lm_weight: вес language model (только для CPU, default: 2.0)
            word_score: бонус за слово (только для CPU, default: 0.0)
            blank_token: токен для blank (default: "<blank>")
            sil_token: токен для silence/space (только для CPU, default: "<blank>")
            unk_word: токен для unknown слов (только для CPU, default: "<unk>")
            nbest: количество лучших гипотез (default: 1)
            blank_skip_threshold: порог для пропуска blank frames (только для CUDA, default: 0.95)
        """
        self.tokens = tokens
        self.blank_token = blank_token
        self.nbest = nbest
        self.has_lexicon = lexicon_path is not None
        
        # ВАЖНО: TorchAudio decoder не поддерживает пустые строки в tokens
        # Заменяем пустую строку на специальный токен <space>
        # 
        # LEGACY SUPPORT: Текущая модель обучена с charset, где пустая строка '' используется как space.
        # После переобучения модели с charset без '' этот код можно упростить до:
        # self.tokens_for_decoder = tokens
        self.tokens_for_decoder = []
        self.space_token_idx = None
        for idx, token in enumerate(tokens):
            if token == '':
                self.tokens_for_decoder.append('<space>')
                self.space_token_idx = idx
            else:
                self.tokens_for_decoder.append(token)
        
        # Определяем, какой decoder использовать
        self.use_cuda = use_cuda and CUDA_DECODER_AVAILABLE and torch.cuda.is_available()
        
        if self.use_cuda:
            # CUDA decoder - упрощённый API
            self.decoder = cuda_ctc_decoder(
                tokens=self.tokens_for_decoder,
                nbest=nbest,
                beam_size=beam_size,
                blank_skip_threshold=blank_skip_threshold
            )
            self.decoder_type = "CUDA"
        else:
            # CPU decoder - полный API
            self.decoder = ctc_decoder(
                lexicon=str(lexicon_path) if lexicon_path else None,
                tokens=self.tokens_for_decoder,
                lm=str(lm_path) if lm_path else None,
                nbest=nbest,
                beam_size=beam_size,
                beam_size_token=beam_size_token,
                beam_threshold=beam_threshold,
                lm_weight=lm_weight,
                word_score=word_score,
                blank_token=blank_token,
                sil_token=sil_token,
                unk_word=unk_word
            )
            self.decoder_type = "CPU"
    
    def decode(self, emissions: torch.Tensor, lengths: torch.Tensor | None = None) -> list[str]:
        """
        Декодирует emissions в текст.

        Args:
            emissions: [time, batch, num_tokens] или [batch, time, num_tokens] - log probabilities от модели
            lengths: [batch] - длины последовательностей (только для CUDA decoder, опционально для CPU)

        Returns:
            список декодированных текстов (по одному на каждый элемент батча)
        """
        # TorchAudio decoder ожидает [batch, time, num_tokens]
        # 
        # ВАЖНО: Мы НЕ можем надёжно определить формат по размерам!
        # Вместо этого предполагаем, что emissions уже в правильном формате [batch, time, num_tokens]
        # Caller должен транспонировать перед вызовом если нужно
        
        # Убеждаемся что tensor contiguous
        if not emissions.is_contiguous():
            emissions = emissions.contiguous()

        if self.decoder_type == "CUDA":
            # CUDA decoder требует encoder_out_lens (длины последовательностей)
            if lengths is None:
                # Если lengths не переданы, используем полную длину для всех элементов батча
                batch_size = emissions.shape[0]
                time_steps = emissions.shape[1]
                lengths = torch.full((batch_size,), time_steps, dtype=torch.int32, device=emissions.device)
            else:
                # Конвертируем в int32 если нужно
                if lengths.dtype != torch.int32:
                    lengths = lengths.to(torch.int32)
                # Перемещаем на то же устройство что emissions
                if lengths.device != emissions.device:
                    lengths = lengths.to(emissions.device)

            # CUDA decoder работает с CUDA тензорами
            if not emissions.is_cuda:
                raise RuntimeError("CUDA decoder requires CUDA tensors")

            # Декодируем
            batch_hypotheses = self.decoder(emissions, lengths)
        else:
            # CPU decoder требует CPU тензоры
            if emissions.is_cuda:
                emissions = emissions.cpu()

            # Декодируем (CPU decoder не требует lengths)
            batch_hypotheses = self.decoder(emissions)

        # Извлекаем лучшие гипотезы
        transcripts = []
        
        # DEBUG: Проверяем количество гипотез
        if len(batch_hypotheses) != emissions.shape[0]:
            print(f"⚠️  WARNING: Decoder returned {len(batch_hypotheses)} hypotheses for batch_size={emissions.shape[0]}")
            print(f"   Emissions shape: {emissions.shape}")
            print(f"   Lengths: {lengths if self.decoder_type == 'CUDA' else 'N/A'}")
            # Обрезаем до правильного размера
            batch_hypotheses = batch_hypotheses[:emissions.shape[0]]
        
        for hypotheses in batch_hypotheses:
            best_hypo = hypotheses[0]

            if self.has_lexicon:
                # Lexicon decoder - используем words (это строки)
                text = " ".join(best_hypo.words)
            else:
                # Lexicon-free decoder - tokens это тензор индексов (CPU) или список (CUDA)
                if isinstance(best_hypo.tokens, torch.Tensor):
                    token_indices = best_hypo.tokens.tolist()
                else:
                    token_indices = best_hypo.tokens

                chars = []
                for idx in token_indices:
                    if idx == 0:  # blank
                        continue
                    # Используем tokens_for_decoder (с <space> вместо пустой строки)
                    char = self.tokens_for_decoder[idx]
                    if char == '<space>':  # конвертируем обратно в пробел
                        chars.append(' ')
                    else:
                        chars.append(char)
                text = "".join(chars)

            transcripts.append(text)

        return transcripts

    
    def decode_with_timestamps(self, emissions: torch.Tensor) -> list[dict]:
        """
        Декодирует emissions с временными метками.
        
        ПРИМЕЧАНИЕ: Этот метод полезен для:
        - Видео OCR (синхронизация с видео)
        - Анализ скорости письма/печати
        - Визуализация процесса распознавания
        
        Args:
            emissions: [time, batch, num_tokens] или [batch, time, num_tokens] - log probabilities от модели
            
        Returns:
            список словарей с полной информацией о декодировании:
            {
                'text': str,
                'tokens': list[str],
                'timesteps': list[int],
                'score': float
            }
        """
        # TorchAudio decoder требует CPU tensor
        if emissions.is_cuda:
            emissions = emissions.cpu()
        
        # Транспонируем если нужно
        if emissions.dim() == 3:
            if emissions.shape[0] < emissions.shape[1]:
                emissions = emissions.transpose(0, 1).contiguous()
        
        # Убеждаемся что tensor contiguous
        if not emissions.is_contiguous():
            emissions = emissions.contiguous()
        
        # Декодируем
        batch_hypotheses = self.decoder(emissions)
        
        # Извлекаем детальную информацию
        results = []
        for hypotheses in batch_hypotheses:
            best_hypo = hypotheses[0]
            
            # Конвертируем token indices в символы
            token_indices = best_hypo.tokens.tolist()
            token_chars = [self.tokens[idx] for idx in token_indices]
            
            if self.has_lexicon:
                text = " ".join(best_hypo.words)
                result = {
                    'text': text,
                    'tokens': token_chars,
                    'words': best_hypo.words,
                    'timesteps': best_hypo.timesteps.tolist(),
                    'score': best_hypo.score
                }
            else:
                text = "".join(token_chars)
                result = {
                    'text': text,
                    'tokens': token_chars,
                    'timesteps': best_hypo.timesteps.tolist(),
                    'score': best_hypo.score
                }
            
            results.append(result)
        
        return results


def create_tokens_file(charset: list[str], output_path: Path | str):
    """
    Создаёт tokens файл для TorchAudio decoder из charset.
    
    Args:
        charset: список символов (включая <blank>)
        output_path: путь для сохранения tokens файла
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in charset:
            f.write(f"{token}\n")
    
    return output_path
