"""
TorchAudio CTC Beam Search Decoder с KenLM language model.

Использует официальную реализацию от PyTorch для быстрого beam search декодирования.
"""
from pathlib import Path
from typing import Optional

import torch
from torchaudio.models.decoder import ctc_decoder


class BeamSearchDecoder:
    """
    Wrapper для TorchAudio CTC beam search decoder.
    
    Поддерживает:
    - Lexicon-free декодирование (без словаря)
    - KenLM n-gram language model
    - Настраиваемые параметры beam search
    """
    
    def __init__(
        self,
        tokens: list[str],
        lm_path: Optional[Path | str] = None,
        lexicon_path: Optional[Path | str] = None,
        beam_size: int = 50,
        beam_size_token: Optional[int] = None,
        beam_threshold: float = 50.0,
        lm_weight: float = 2.0,
        word_score: float = 0.0,
        blank_token: str = "<blank>",
        sil_token: str = "<blank>",  # Используем blank как silence (нет отдельного sil токена)
        unk_word: str = "<unk>",
        nbest: int = 1
    ):
        """
        Инициализация beam search decoder.
        
        Args:
            tokens: список токенов (charset) в том же порядке, что и в модели
            lm_path: путь к KenLM language model (.arpa или .bin)
            lexicon_path: путь к lexicon файлу (опционально)
            beam_size: размер beam для декодирования (default: 50)
            beam_size_token: размер beam на уровне токенов (default: None = beam_size)
            beam_threshold: порог для pruning (default: 50.0)
            lm_weight: вес language model (alpha) (default: 2.0)
            word_score: бонус за слово (beta) (default: 0.0)
            blank_token: токен для blank (default: "<blank>")
            sil_token: токен для silence/space (default: "<blank>" - используем blank)
            unk_word: токен для unknown слов (default: "<unk>")
            nbest: количество лучших гипотез (default: 1)
        """
        self.tokens = tokens
        self.blank_token = blank_token
        self.nbest = nbest
        self.has_lexicon = lexicon_path is not None
        
        # ВАЖНО: TorchAudio decoder не поддерживает пустые строки в tokens
        # Заменяем пустую строку на специальный токен <space>
        self.tokens_for_decoder = []
        self.space_token_idx = None
        for idx, token in enumerate(tokens):
            if token == '':
                self.tokens_for_decoder.append('<space>')
                self.space_token_idx = idx
            else:
                self.tokens_for_decoder.append(token)
        
        # Создаём decoder через фабричную функцию
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
    
    def decode(self, emissions: torch.Tensor) -> list[str]:
        """
        Декодирует emissions в текст.
        
        Args:
            emissions: [time, batch, num_tokens] или [batch, time, num_tokens] - log probabilities от модели
            
        Returns:
            список декодированных текстов (по одному на каждый элемент батча)
        """
        # TorchAudio decoder требует CPU tensor
        if emissions.is_cuda:
            emissions = emissions.cpu()
        
        # TorchAudio decoder ожидает [batch, time, num_tokens]
        # Наша модель возвращает [time, batch, num_tokens]
        # Проверяем и транспонируем если нужно
        if emissions.dim() == 3:
            if emissions.shape[0] < emissions.shape[1]:
                # Похоже на [time, batch, num_tokens] - транспонируем
                emissions = emissions.transpose(0, 1).contiguous()  # [batch, time, num_tokens]
        
        # Убеждаемся что tensor contiguous
        if not emissions.is_contiguous():
            emissions = emissions.contiguous()
        
        # Декодируем
        batch_hypotheses = self.decoder(emissions)
        
        # Извлекаем лучшие гипотезы
        transcripts = []
        for hypotheses in batch_hypotheses:
            best_hypo = hypotheses[0]
            
            if self.has_lexicon:
                # Lexicon decoder - используем words (это строки)
                text = " ".join(best_hypo.words)
            else:
                # Lexicon-free decoder - tokens это тензор индексов, конвертируем в символы
                token_indices = best_hypo.tokens.tolist()
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
