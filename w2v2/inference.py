"""Inference module for phoneme recognition with timestamps using CTC segmentation."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

try:
    import ctc_segmentation
    CTC_SEGMENTATION_AVAILABLE = True
except ImportError:
    CTC_SEGMENTATION_AVAILABLE = False

from w2v2.phoneme import text_to_phoneme_list, devanagari_to_ipa, ipa_to_phoneme_list
from w2v2.data import get_sample_audio, load_vocabulary
from lib.logger import console, info, success, error, warning, print_phonemes


def _get_default_device() -> str:
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_path: str = "outputs/phoneme-model"
    device: str = None  # Auto-detect if None
    sampling_rate: int = 16000
    # CTC segmentation parameters
    blank_token_id: int = 0
    frame_duration_ms: float = 20.0  # wav2vec2 outputs at ~50fps = 20ms per frame
    
    def __post_init__(self):
        if self.device is None:
            self.device = _get_default_device()


class PhonemeRecognizer:
    """Hindi speech to phoneme recognizer with timestamps."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        info(f"Loading model from {config.model_path}...")
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(config.model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(config.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load vocabulary for decoding
        vocab_path = os.path.join(config.model_path, "vocab.json")
        self.vocab, self.id_to_phoneme = load_vocabulary(vocab_path)
        
        success(f"Model loaded on {config.device}")
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.config.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.config.sampling_rate
            )
            waveform = resampler(waveform)
        
        return waveform.squeeze()
    
    def get_logits(self, waveform: torch.Tensor) -> np.ndarray:
        """Run model inference to get frame-level logits."""
        # Prepare input
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        
        input_values = inputs.input_values.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_values)
            logits = outputs.logits
        
        # Convert to log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        return log_probs.cpu().numpy()[0]
    
    def greedy_decode(self, logits: np.ndarray) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Greedy CTC decoding with timestamps.
        Returns phonemes and their (start, end) timestamps in seconds.
        """
        # Get predicted token IDs
        pred_ids = np.argmax(logits, axis=-1)
        
        # Debug: show distribution of predicted tokens
        unique, counts = np.unique(pred_ids, return_counts=True)
        token_dist = {self.id_to_phoneme.get(int(u), f"id={u}"): int(c) for u, c in zip(unique, counts)}
        info(f"Token distribution: {token_dist}")
        
        phonemes = []
        timestamps = []
        
        frame_duration = self.config.frame_duration_ms / 1000.0  # Convert to seconds
        
        prev_token = None
        start_frame = 0
        
        for frame_idx, token_id in enumerate(pred_ids):
            # Skip blank tokens and repeated tokens
            if token_id == self.config.blank_token_id:
                if prev_token is not None and prev_token != self.config.blank_token_id:
                    # End of current phoneme
                    end_time = frame_idx * frame_duration
                    start_time = start_frame * frame_duration
                    
                    phoneme = self.id_to_phoneme.get(prev_token, "<unk>")
                    if phoneme not in ["<pad>", "<unk>", "|"]:
                        phonemes.append(phoneme)
                        timestamps.append((start_time, end_time))
                
                prev_token = token_id
                continue
            
            if token_id != prev_token:
                # New phoneme starts
                if prev_token is not None and prev_token != self.config.blank_token_id:
                    # End previous phoneme
                    end_time = frame_idx * frame_duration
                    start_time = start_frame * frame_duration
                    
                    phoneme = self.id_to_phoneme.get(prev_token, "<unk>")
                    if phoneme not in ["<pad>", "<unk>", "|"]:
                        phonemes.append(phoneme)
                        timestamps.append((start_time, end_time))
                
                start_frame = frame_idx
            
            prev_token = token_id
        
        # Handle last phoneme
        if prev_token is not None and prev_token != self.config.blank_token_id:
            end_time = len(pred_ids) * frame_duration
            start_time = start_frame * frame_duration
            
            phoneme = self.id_to_phoneme.get(prev_token, "<unk>")
            if phoneme not in ["<pad>", "<unk>", "|"]:
                phonemes.append(phoneme)
                timestamps.append((start_time, end_time))
        
        return phonemes, timestamps
    
    def ctc_segment(
        self, 
        logits: np.ndarray, 
        text: Optional[str] = None,
        phoneme_sequence: Optional[list[str]] = None,
    ) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Use CTC segmentation for more accurate timestamps.
        Requires either text (Devanagari) or a phoneme sequence.
        """
        if not CTC_SEGMENTATION_AVAILABLE:
            warning("ctc_segmentation not available, falling back to greedy decode")
            return self.greedy_decode(logits)
        
        # Get phoneme sequence
        if phoneme_sequence is None:
            if text is None:
                # Use greedy decoding to get phonemes first
                phonemes, _ = self.greedy_decode(logits)
                phoneme_sequence = phonemes
            else:
                # Convert text to phonemes
                phoneme_sequence = text_to_phoneme_list(text)
        
        if not phoneme_sequence:
            return [], []
        
        # Prepare for CTC segmentation
        frame_duration = self.config.frame_duration_ms / 1000.0
        
        # Create character dictionary for segmentation
        char_list = list(self.vocab.keys())
        
        # Configure CTC segmentation
        config = ctc_segmentation.CtcSegmentationParameters(
            char_list=char_list,
            blank=self.config.blank_token_id,
        )
        config.index_duration = frame_duration
        
        # Prepare ground truth: join phonemes with space
        ground_truth = " ".join(phoneme_sequence)
        
        try:
            # Run CTC segmentation
            timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
                config, logits, [ground_truth]
            )
            
            # Parse results
            phonemes = []
            timestamps = []
            
            segments = timings[0]  # First (and only) utterance
            
            for i, phoneme in enumerate(phoneme_sequence):
                if i < len(segments):
                    start_time, end_time, confidence = segments[i]
                    phonemes.append(phoneme)
                    timestamps.append((start_time, end_time))
            
            return phonemes, timestamps
            
        except Exception as e:
            warning(f"CTC segmentation failed: {e}, falling back to greedy decode")
            return self.greedy_decode(logits)
    
    def recognize(
        self, 
        audio_path: str,
        reference_text: Optional[str] = None,
        use_ctc_segmentation: bool = True,
    ) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Recognize phonemes with timestamps from audio file.
        
        Args:
            audio_path: Path to audio file
            reference_text: Optional Devanagari text for CTC segmentation
            use_ctc_segmentation: Whether to use CTC segmentation (more accurate)
        
        Returns:
            Tuple of (phonemes, timestamps) where timestamps are (start, end) in seconds
        """
        info(f"Processing: {audio_path}")
        
        # Load audio
        waveform = self.load_audio(audio_path)
        audio_duration = len(waveform) / self.config.sampling_rate
        info(f"Audio duration: {audio_duration:.2f}s")
        
        # Get model predictions
        logits = self.get_logits(waveform)
        info(f"Got {logits.shape[0]} frames from model")
        
        # Decode with timestamps
        if use_ctc_segmentation and reference_text:
            phonemes, timestamps = self.ctc_segment(logits, text=reference_text)
        else:
            phonemes, timestamps = self.greedy_decode(logits)
        
        success(f"Recognized {len(phonemes)} phonemes")
        
        return phonemes, timestamps


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(
    phonemes: list[str],
    timestamps: list[tuple[float, float]],
    output_path: str,
    min_duration: float = 0.01,  # Minimum duration for a subtitle
):
    """Write phonemes with timestamps to SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        subtitle_idx = 1
        
        for phoneme, (start, end) in zip(phonemes, timestamps):
            # Ensure minimum duration
            if end - start < min_duration:
                end = start + min_duration
            
            # Write SRT entry
            f.write(f"{subtitle_idx}\n")
            f.write(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n")
            f.write(f"{phoneme}\n")
            f.write("\n")
            
            subtitle_idx += 1
    
    success(f"SRT file written to {output_path}")


def transcribe_audio(
    audio_path: str,
    model_path: str = "outputs/phoneme-model",
    output_srt: Optional[str] = None,
    reference_text: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[list[str], list[tuple[float, float]]]:
    """
    Main function to transcribe audio to timestamped phonemes.
    
    Args:
        audio_path: Path to input audio file
        model_path: Path to trained model
        output_srt: Path for output SRT file (default: same as audio with .srt extension)
        reference_text: Optional Devanagari text for better alignment
        device: Device to use (cuda/mps/cpu, auto-detects if None)
    
    Returns:
        Tuple of (phonemes, timestamps)
    """
    # Setup config (device auto-detected if None)
    config = InferenceConfig(
        model_path=model_path,
        device=device,  # Let InferenceConfig auto-detect if None
    )
    
    # Create recognizer
    recognizer = PhonemeRecognizer(config)
    
    # Recognize
    phonemes, timestamps = recognizer.recognize(
        audio_path,
        reference_text=reference_text,
        use_ctc_segmentation=reference_text is not None,
    )
    
    # Print results
    print_phonemes(phonemes, timestamps, title="Recognized Phonemes")
    
    # Write SRT
    if output_srt is None:
        audio_path_obj = Path(audio_path)
        output_srt = str(audio_path_obj.with_suffix(".srt"))
    
    write_srt(phonemes, timestamps, output_srt)
    
    return phonemes, timestamps


class BaseModelTranscriber:
    """Transcriber using base wav2vec2-xlsr model for Hindi ASR."""
    
    DEFAULT_MODEL = "theainerd/Wav2Vec2-large-xlsr-hindi"
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        
        # Auto-detect device: CUDA > MPS > CPU
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.sampling_rate = 16000
        self.frame_duration_ms = 20.0  # wav2vec2 outputs at ~50fps = 20ms per frame
        
        info(f"Loading base model: {self.model_name}...")
        
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        success(f"Model loaded on {self.device}")
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze()
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to Hindi text."""
        info(f"Processing: {audio_path}")
        
        # Load audio
        waveform = self.load_audio(audio_path)
        audio_duration = len(waveform) / self.sampling_rate
        info(f"Audio duration: {audio_duration:.2f}s")
        
        # Prepare input
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        
        input_values = inputs.input_values.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_values)
            logits = outputs.logits
        
        # Decode
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        
        success(f"Transcription complete")
        
        return transcription
    
    def transcribe_with_timestamps(
        self, 
        audio_path: str,
        word_level: bool = True,
        word_gap_threshold: float = 0.12,
    ) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Transcribe audio file to Hindi text with timestamps.
        
        Args:
            audio_path: Path to audio file
            word_level: If True, return word-level timestamps. If False, return character-level.
            word_gap_threshold: Time gap (in seconds) between characters to detect word boundaries.
                               Lower values = more word splits. Default: 0.12s (120ms)
        
        Returns:
            Tuple of (words/chars, timestamps) where timestamps are (start, end) in seconds
        """
        info(f"Processing: {audio_path}")
        
        # Load audio
        waveform = self.load_audio(audio_path)
        audio_duration = len(waveform) / self.sampling_rate
        info(f"Audio duration: {audio_duration:.2f}s")
        
        # Prepare input
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        
        input_values = inputs.input_values.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_values)
            logits = outputs.logits
        
        # Get predicted token IDs
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
        # Get vocabulary for decoding
        vocab = self.processor.tokenizer.get_vocab()
        id_to_char = {v: k for k, v in vocab.items()}
        
        # Find blank token (usually <pad> or index 0)
        blank_id = self.processor.tokenizer.pad_token_id or 0
        
        # Frame duration in seconds
        frame_duration = self.frame_duration_ms / 1000.0
        
        # Extract characters with timestamps using CTC collapse
        chars = []
        char_timestamps = []
        
        prev_token = None
        start_frame = 0
        
        for frame_idx, token_id in enumerate(pred_ids):
            token_id = int(token_id)
            
            # Skip blank tokens
            if token_id == blank_id:
                if prev_token is not None and prev_token != blank_id:
                    # End of current character
                    end_time = frame_idx * frame_duration
                    start_time = start_frame * frame_duration
                    
                    char = id_to_char.get(prev_token, "")
                    if char and char not in ["<pad>", "<unk>", "<s>", "</s>", "|"]:
                        # Handle special tokens like "|" as word boundary
                        chars.append(char)
                        char_timestamps.append((start_time, end_time))
                
                prev_token = token_id
                continue
            
            if token_id != prev_token:
                # New character starts
                if prev_token is not None and prev_token != blank_id:
                    # End previous character
                    end_time = frame_idx * frame_duration
                    start_time = start_frame * frame_duration
                    
                    char = id_to_char.get(prev_token, "")
                    if char and char not in ["<pad>", "<unk>", "<s>", "</s>", "|"]:
                        chars.append(char)
                        char_timestamps.append((start_time, end_time))
                
                start_frame = frame_idx
            
            prev_token = token_id
        
        # Handle last character
        if prev_token is not None and prev_token != blank_id:
            end_time = len(pred_ids) * frame_duration
            start_time = start_frame * frame_duration
            
            char = id_to_char.get(prev_token, "")
            if char and char not in ["<pad>", "<unk>", "<s>", "</s>", "|"]:
                chars.append(char)
                char_timestamps.append((start_time, end_time))
        
        if not word_level:
            success(f"Recognized {len(chars)} characters")
            return chars, char_timestamps
        
        # Group into words based on timing gaps
        # When there's a gap > threshold between characters, treat as word boundary
        
        words = []
        word_timestamps = []
        
        current_word = []
        word_start = None
        word_end = None
        prev_end = None
        
        for char, (start, end) in zip(chars, char_timestamps):
            # Check for explicit word boundary tokens
            if char == " " or char == "|":
                if current_word:
                    word_text = "".join(current_word)
                    words.append(word_text)
                    word_timestamps.append((word_start, word_end))
                    current_word = []
                    word_start = None
                    word_end = None
                prev_end = end
                continue
            
            # Check for timing gap (word boundary detection)
            if prev_end is not None and (start - prev_end) > word_gap_threshold:
                # Gap detected - save current word and start new one
                if current_word:
                    word_text = "".join(current_word)
                    words.append(word_text)
                    word_timestamps.append((word_start, word_end))
                    current_word = []
                    word_start = None
                    word_end = None
            
            # Add character to current word
            if word_start is None:
                word_start = start
            word_end = end
            current_word.append(char)
            prev_end = end
        
        # Don't forget the last word
        if current_word:
            word_text = "".join(current_word)
            words.append(word_text)
            word_timestamps.append((word_start, word_end))
        
        success(f"Recognized {len(words)} words")
        
        return words, word_timestamps


def transcribe_with_base_model(
    audio_path: str,
    model_name: str = None,
    device: str = None,
) -> str:
    """
    Transcribe audio to Hindi text using the base wav2vec2-xlsr model.
    
    Args:
        audio_path: Path to input audio file
        model_name: HuggingFace model name (default: theainerd/Wav2Vec2-large-xlsr-hindi)
        device: Device to use (cuda/cpu)
    
    Returns:
        Transcribed Hindi text
    """
    transcriber = BaseModelTranscriber(model_name=model_name, device=device)
    transcription = transcriber.transcribe(audio_path)
    
    return transcription


def write_hindi_srt(
    words: list[str],
    timestamps: list[tuple[float, float]],
    output_path: str,
    min_duration: float = 0.05,
    group_words: int = 0,
):
    """
    Write Hindi words with timestamps to SRT file.
    
    Args:
        words: List of Hindi words
        timestamps: List of (start, end) timestamps in seconds
        output_path: Output SRT file path
        min_duration: Minimum duration for a subtitle
        group_words: If > 0, group this many words per subtitle entry (0 = one word per entry)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        subtitle_idx = 1
        
        if group_words > 0:
            # Group words together
            for i in range(0, len(words), group_words):
                group = words[i:i + group_words]
                group_timestamps = timestamps[i:i + group_words]
                
                text = " ".join(group)
                start = group_timestamps[0][0]
                end = group_timestamps[-1][1]
                
                # Ensure minimum duration
                if end - start < min_duration:
                    end = start + min_duration
                
                f.write(f"{subtitle_idx}\n")
                f.write(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n")
                f.write(f"{text}\n")
                f.write("\n")
                
                subtitle_idx += 1
        else:
            # One word per entry
            for word, (start, end) in zip(words, timestamps):
                # Ensure minimum duration
                if end - start < min_duration:
                    end = start + min_duration
                
                f.write(f"{subtitle_idx}\n")
                f.write(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n")
                f.write(f"{word}\n")
                f.write("\n")
                
                subtitle_idx += 1
    
    success(f"Hindi SRT file written to {output_path}")


def transcribe_hindi_to_srt(
    audio_path: str,
    output_srt: Optional[str] = None,
    model_name: str = None,
    device: str = None,
    group_words: int = 5,
    char_level: bool = False,
) -> tuple[list[str], list[tuple[float, float]]]:
    """
    Transcribe audio to Hindi text with timestamps and save as SRT.
    
    Args:
        audio_path: Path to input audio file
        output_srt: Path for output SRT file (default: same as audio with .srt extension)
        model_name: HuggingFace model name (default: theainerd/Wav2Vec2-large-xlsr-hindi)
        device: Device to use (cuda/cpu)
        group_words: Number of words to group per subtitle entry (default: 5, ignored if char_level=True)
        char_level: If True, output each character with its timestamp (one per SRT entry)
    
    Returns:
        Tuple of (characters/words, timestamps)
    """
    transcriber = BaseModelTranscriber(model_name=model_name, device=device)
    
    # Get character or word level timestamps
    items, timestamps = transcriber.transcribe_with_timestamps(
        audio_path, 
        word_level=not char_level
    )
    
    # Print results
    if char_level:
        console.print("\n[bold cyan]Transcribed Characters:[/bold cyan]")
        # Show characters in a more compact format
        for i in range(0, len(items), 20):
            chunk = items[i:i+20]
            chunk_ts = timestamps[i:i+20]
            chars_str = " ".join(f"{c}({s:.2f})" for c, (s, e) in zip(chunk, chunk_ts))
            console.print(f"  {chars_str}")
    else:
        console.print("\n[bold cyan]Transcribed Words:[/bold cyan]")
        for word, (start, end) in zip(items, timestamps):
            console.print(f"  [{start:.2f}s - {end:.2f}s] {word}")
    
    # Determine output path
    if output_srt is None:
        audio_path_obj = Path(audio_path)
        output_srt = str(audio_path_obj.with_suffix(".srt"))
    
    # Write SRT - for char_level, each character gets its own entry
    if char_level:
        write_hindi_srt(items, timestamps, output_srt, group_words=0)
    else:
        write_hindi_srt(items, timestamps, output_srt, group_words=group_words)
    
    return items, timestamps


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        error("Usage: python -m w2v2.inference <audio_file> [model_path]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/phoneme-model"
    
    transcribe_audio(audio_path, model_path)
