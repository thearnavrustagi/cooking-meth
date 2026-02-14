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
from w2v2.data import get_sample_audio, load_vocabulary, prepare_dataset, PhonemeDataCollator
from lib.logger import console, info, success, error, warning, print_phonemes, print_results


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
    frame_duration_ms: float = None  # Auto-detect from model if None
    
    def __post_init__(self):
        if self.device is None:
            self.device = _get_default_device()
        # frame_duration_ms will be set by PhonemeRecognizer based on model config


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
        
        # Auto-detect frame duration from model config
        if config.frame_duration_ms is None:
            # Calculate total stride from conv_stride
            total_stride = 1
            for s in self.model.config.conv_stride:
                total_stride *= s
            # At 16kHz, each sample is 1/16000 seconds = 0.0625ms
            # frame_duration_ms = total_stride * (1000 / sampling_rate)
            self.config.frame_duration_ms = total_stride / 16.0  # 16 samples per ms at 16kHz
            info(f"Auto-detected frame duration: {self.config.frame_duration_ms}ms (stride={total_stride})")
        
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
    
    def greedy_decode(self, logits: np.ndarray, silence_threshold_ms: float = 200.0) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Greedy CTC decoding with timestamps.
        Returns phonemes and their (start, end) timestamps in seconds.
        
        Key insight: In CTC, blank tokens are interspersed with phoneme predictions.
        We should NOT end a segment when we see a blank - only when:
        1. A DIFFERENT non-blank phoneme appears
        2. There's a long silence (many consecutive blanks exceeding silence_threshold_ms)
        
        Args:
            logits: Frame-level log probabilities from the model
            silence_threshold_ms: End segment if blanks exceed this duration (default 200ms)
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
        silence_threshold_frames = int(silence_threshold_ms / self.config.frame_duration_ms)
        
        # Track current phoneme segment
        current_phoneme_id = None  # The active non-blank phoneme
        current_phoneme = None
        start_frame = None
        last_phoneme_frame = None  # Last frame where we saw the current phoneme (not blank)
        blank_count = 0  # Consecutive blank frames
        
        for frame_idx, token_id in enumerate(pred_ids):
            is_blank = (token_id == self.config.blank_token_id)
            
            if is_blank:
                blank_count += 1
                
                # Check for silence (long blank span) - end current segment
                if current_phoneme is not None and blank_count >= silence_threshold_frames:
                    # End the segment at the last phoneme frame
                    end_time = (last_phoneme_frame + 1) * frame_duration
                    start_time = start_frame * frame_duration
                    
                    if current_phoneme not in ["<pad>", "<unk>", "|"]:
                        phonemes.append(current_phoneme)
                        timestamps.append((start_time, end_time))
                    
                    current_phoneme_id = None
                    current_phoneme = None
                    start_frame = None
                    last_phoneme_frame = None
                
                continue
            
            # Non-blank token
            phoneme = self.id_to_phoneme.get(token_id, "<unk>")
            
            # Skip special tokens
            if phoneme in ["<pad>", "<unk>", "|"]:
                continue
            
            # Reset blank counter since we saw a non-blank
            blank_count = 0
            
            if current_phoneme is None:
                # Start new segment
                current_phoneme_id = token_id
                current_phoneme = phoneme
                start_frame = frame_idx
                last_phoneme_frame = frame_idx
                
            elif token_id == current_phoneme_id:
                # Same phoneme continues - extend the segment
                last_phoneme_frame = frame_idx
                
            else:
                # DIFFERENT phoneme - end current segment and start new one
                end_time = (last_phoneme_frame + 1) * frame_duration
                start_time = start_frame * frame_duration
                
                if current_phoneme not in ["<pad>", "<unk>", "|"]:
                    phonemes.append(current_phoneme)
                    timestamps.append((start_time, end_time))
                
                # Start new segment
                current_phoneme_id = token_id
                current_phoneme = phoneme
                start_frame = frame_idx
                last_phoneme_frame = frame_idx
        
        # Handle last phoneme
        if current_phoneme is not None:
            end_time = (last_phoneme_frame + 1) * frame_duration
            start_time = start_frame * frame_duration
            
            if current_phoneme not in ["<pad>", "<unk>", "|"]:
                phonemes.append(current_phoneme)
                timestamps.append((start_time, end_time))
        
        return phonemes, timestamps
    
    def extend_to_fill_gaps(
        self,
        phonemes: list[str],
        timestamps: list[tuple[float, float]],
        audio_duration: float,
    ) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Post-process timestamps to extend each phoneme until the next one starts.
        
        CTC models often output "spiky" predictions where phonemes only last 1-2 frames.
        This function extends each phoneme's duration to fill the gap until the next phoneme,
        which is more representative of actual speech where sounds are continuous.
        
        Args:
            phonemes: List of phoneme strings
            timestamps: List of (start, end) tuples
            audio_duration: Total audio duration in seconds
        
        Returns:
            Same phonemes with extended timestamps
        """
        if not phonemes:
            return phonemes, timestamps
        
        extended_timestamps = []
        
        for i, (phoneme, (start, end)) in enumerate(zip(phonemes, timestamps)):
            # Extend end time to the start of the next phoneme
            if i < len(phonemes) - 1:
                next_start = timestamps[i + 1][0]
                new_end = next_start
            else:
                # Last phoneme extends to end of audio
                new_end = audio_duration
            
            extended_timestamps.append((start, new_end))
        
        return phonemes, extended_timestamps
    
    def greedy_decode_raw(self, logits: np.ndarray) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Raw frame-level decoding WITHOUT any CTC collapsing.
        
        Returns every frame's prediction as a separate entry, showing exactly
        what the model outputs including blanks and repeated tokens.
        
        Useful for debugging and understanding model behavior.
        """
        # Get predicted token IDs
        pred_ids = np.argmax(logits, axis=-1)
        
        # Debug: show distribution of predicted tokens
        unique, counts = np.unique(pred_ids, return_counts=True)
        token_dist = {self.id_to_phoneme.get(int(u), f"id={u}"): int(c) for u, c in zip(unique, counts)}
        info(f"Token distribution: {token_dist}")
        info(f"Total frames: {len(pred_ids)}")
        
        phonemes = []
        timestamps = []
        
        frame_duration = self.config.frame_duration_ms / 1000.0  # Convert to seconds
        
        # Group consecutive identical tokens into spans (but don't collapse different tokens)
        current_token_id = None
        current_phoneme = None
        start_frame = 0
        
        for frame_idx, token_id in enumerate(pred_ids):
            phoneme = self.id_to_phoneme.get(int(token_id), f"<id={token_id}>")
            
            # Replace special tokens with readable names
            if token_id == self.config.blank_token_id:
                phoneme = "_"  # Use underscore for blank/pad
            
            if token_id != current_token_id:
                # Save previous span
                if current_phoneme is not None:
                    start_time = start_frame * frame_duration
                    end_time = frame_idx * frame_duration
                    phonemes.append(current_phoneme)
                    timestamps.append((start_time, end_time))
                
                # Start new span
                current_token_id = token_id
                current_phoneme = phoneme
                start_frame = frame_idx
        
        # Handle last span
        if current_phoneme is not None:
            start_time = start_frame * frame_duration
            end_time = len(pred_ids) * frame_duration
            phonemes.append(current_phoneme)
            timestamps.append((start_time, end_time))
        
        return phonemes, timestamps
    
    def recognize_raw(
        self,
        audio_path: str,
    ) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Get raw frame-level predictions without CTC collapsing.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (tokens, timestamps) showing exact model output
        """
        info(f"Processing (raw mode): {audio_path}")
        
        # Load audio
        waveform = self.load_audio(audio_path)
        audio_duration = len(waveform) / self.config.sampling_rate
        info(f"Audio duration: {audio_duration:.2f}s")
        
        # Get model predictions
        logits = self.get_logits(waveform)
        info(f"Got {logits.shape[0]} frames from model ({self.config.frame_duration_ms}ms each)")
        
        # Decode without CTC collapsing
        tokens, timestamps = self.greedy_decode_raw(logits)
        
        return tokens, timestamps
    
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
        extend_timestamps: bool = True,
    ) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Recognize phonemes with timestamps from audio file.
        
        Args:
            audio_path: Path to audio file
            reference_text: Optional Devanagari text for CTC segmentation
            use_ctc_segmentation: Whether to use CTC segmentation (more accurate)
            extend_timestamps: Extend each phoneme to fill gap until next phoneme
        
        Returns:
            Tuple of (phonemes, timestamps) where timestamps are (start, end) in seconds
        """
        info(f"Processing: {audio_path}")
        
        # Load audio
        waveform = self.load_audio(audio_path)
        return self.recognize_waveform(waveform, reference_text, use_ctc_segmentation, extend_timestamps)

    def recognize_waveform(
        self,
        waveform: torch.Tensor,
        reference_text: Optional[str] = None,
        use_ctc_segmentation: bool = True,
        extend_timestamps: bool = True,
    ) -> tuple[list[str], list[tuple[float, float]]]:
        """Recognize phonemes from a waveform tensor."""
        audio_duration = len(waveform) / self.config.sampling_rate
        # info(f"Audio duration: {audio_duration:.2f}s")
        
        # Get model predictions
        logits = self.get_logits(waveform)
        # info(f"Got {logits.shape[0]} frames from model")
        
        # Decode with timestamps
        if use_ctc_segmentation and reference_text:
            phonemes, timestamps = self.ctc_segment(logits, text=reference_text)
        else:
            phonemes, timestamps = self.greedy_decode(logits)
        
        # Extend timestamps to fill gaps (recommended for continuous speech)
        if extend_timestamps and phonemes:
            phonemes, timestamps = self.extend_to_fill_gaps(phonemes, timestamps, audio_duration)
        
        # success(f"Recognized {len(phonemes)} phonemes")
        
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
    raw_output: bool = False,
    extend_timestamps: bool = True,
) -> tuple[list[str], list[tuple[float, float]]]:
    """
    Main function to transcribe audio to timestamped phonemes.
    
    Args:
        audio_path: Path to input audio file
        model_path: Path to trained model
        output_srt: Path for output SRT file (default: same as audio with .srt extension)
        reference_text: Optional Devanagari text for better alignment
        device: Device to use (cuda/mps/cpu, auto-detects if None)
        raw_output: If True, output raw frame-level predictions without CTC collapsing
        extend_timestamps: Extend each phoneme to fill gap until next phoneme (default True)
    
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
    if raw_output:
        phonemes, timestamps = recognizer.recognize_raw(audio_path)
        title = "Raw Model Output (no CTC collapse)"
    else:
        phonemes, timestamps = recognizer.recognize(
            audio_path,
            reference_text=reference_text,
            use_ctc_segmentation=reference_text is not None,
            extend_timestamps=extend_timestamps,
        )
        title = "Recognized Phonemes" + (" (extended)" if extend_timestamps else "")
    
    # Print results
    print_phonemes(phonemes, timestamps, title=title)
    
    # Write SRT
    if output_srt is None:
        audio_path_obj = Path(audio_path)
        suffix = ".raw.srt" if raw_output else ".srt"
        output_srt = str(audio_path_obj.with_suffix(suffix))
    
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
        silence_threshold_ms: float = 200.0,
    ) -> tuple[list[str], list[tuple[float, float]]]:
        """
        Transcribe audio file to Hindi text with timestamps.
        
        Args:
            audio_path: Path to audio file
            word_level: If True, return word-level timestamps. If False, return character-level.
            word_gap_threshold: Time gap (in seconds) between characters to detect word boundaries.
                               Lower values = more word splits. Default: 0.12s (120ms)
            silence_threshold_ms: End segment if blanks exceed this duration (default 200ms)
        
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
        silence_threshold_frames = int(silence_threshold_ms / self.frame_duration_ms)
        
        # Extract characters with timestamps using improved CTC decoding
        # Key fix: Don't end segment on blanks, only on DIFFERENT characters or long silence
        chars = []
        char_timestamps = []
        
        current_char_id = None
        current_char = None
        start_frame = None
        last_char_frame = None
        blank_count = 0
        
        for frame_idx, token_id in enumerate(pred_ids):
            token_id = int(token_id)
            is_blank = (token_id == blank_id)
            
            if is_blank:
                blank_count += 1
                
                # Check for silence (long blank span) - end current segment
                if current_char is not None and blank_count >= silence_threshold_frames:
                    end_time = (last_char_frame + 1) * frame_duration
                    start_time = start_frame * frame_duration
                    
                    if current_char and current_char not in ["<pad>", "<unk>", "<s>", "</s>", "|"]:
                        chars.append(current_char)
                        char_timestamps.append((start_time, end_time))
                    
                    current_char_id = None
                    current_char = None
                    start_frame = None
                    last_char_frame = None
                
                continue
            
            # Non-blank token
            char = id_to_char.get(token_id, "")
            
            # Skip special tokens
            if not char or char in ["<pad>", "<unk>", "<s>", "</s>", "|"]:
                continue
            
            # Reset blank counter
            blank_count = 0
            
            if current_char is None:
                # Start new segment
                current_char_id = token_id
                current_char = char
                start_frame = frame_idx
                last_char_frame = frame_idx
                
            elif token_id == current_char_id:
                # Same character continues - extend the segment
                last_char_frame = frame_idx
                
            else:
                # DIFFERENT character - end current segment and start new one
                end_time = (last_char_frame + 1) * frame_duration
                start_time = start_frame * frame_duration
                
                if current_char and current_char not in ["<pad>", "<unk>", "<s>", "</s>", "|"]:
                    chars.append(current_char)
                    char_timestamps.append((start_time, end_time))
                
                # Start new segment
                current_char_id = token_id
                current_char = char
                start_frame = frame_idx
                last_char_frame = frame_idx
        
        # Handle last character
        if current_char is not None:
            end_time = (last_char_frame + 1) * frame_duration
            start_time = start_frame * frame_duration
            
            if current_char and current_char not in ["<pad>", "<unk>", "<s>", "</s>", "|"]:
                chars.append(current_char)
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


def _evaluate_split(
    recognizer: PhonemeRecognizer,
    split_data,
    split_name: str,
    streaming: bool = False,
):
    """Evaluate model on a single split and return metrics."""
    import evaluate
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from collections import Counter
    
    info(f"Evaluating on {len(split_data)} {split_name} samples...")
    
    wer_metric = evaluate.load("wer")
    
    all_preds = []
    all_labels = []

    # Use DataLoader for batching if not streaming
    if not streaming:
        dataloader = DataLoader(split_data, batch_size=1, shuffle=False)
    else:
        dataloader = split_data

    # Live metrics tracking
    running_per = 0.0
    running_f1 = 0.0
    
    pbar = tqdm(dataloader, desc=f"Evaluating {split_name}")
    skipped = 0
    for batch in pbar:
        # batch is a dict with 'input_values' and 'labels'
        if not isinstance(batch, dict):
            skipped += 1
            continue
        
        input_values = batch.get("input_values")
        label_ids = batch.get("labels")
        
        # Skip invalid/corrupted samples
        if input_values is None or label_ids is None:
            skipped += 1
            continue
        
        # DataLoader wraps in extra dimension - unwrap if needed
        if isinstance(input_values, torch.Tensor):
            if input_values.dim() > 1:
                input_values = input_values.squeeze(0)
        elif isinstance(input_values, list) and len(input_values) > 0 and isinstance(input_values[0], list):
            input_values = input_values[0]
        
        if len(input_values) == 0:
            skipped += 1
            continue
            
        input_values = torch.tensor(input_values) if not isinstance(input_values, torch.Tensor) else input_values
        
        # Unwrap label_ids if needed (DataLoader batching)
        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.squeeze().tolist()  # squeeze all singleton dims
        
        # Handle list of tensors (from default collate function)
        if isinstance(label_ids, list) and len(label_ids) > 0 and isinstance(label_ids[0], torch.Tensor):
            label_ids = [t.item() if t.numel() == 1 else t.tolist() for t in label_ids]
            # Flatten if any element became a list
            if isinstance(label_ids[0], list):
                label_ids = [item for sublist in label_ids for item in sublist]
        
        # Unwrap nested lists until we get a flat list of integers
        while isinstance(label_ids, list) and len(label_ids) > 0 and isinstance(label_ids[0], (list, np.ndarray)):
            label_ids = label_ids[0]
        
        # Ensure we have a flat list of ints (handle numpy arrays too)
        if isinstance(label_ids, np.ndarray):
            label_ids = label_ids.flatten().tolist()

        # Get predictions
        with torch.no_grad():
            logits = recognizer.get_logits(input_values)
            pred_ids = np.argmax(logits, axis=-1)
            
            # Decode for PER
            pred_str = recognizer.processor.batch_decode([pred_ids])[0]
            label_str = recognizer.processor.decode(label_ids, group_tokens=False)
            
            all_preds.append(pred_str)
            all_labels.append(label_str)
            
            # Calculate live metrics every 10 samples
            if len(all_preds) % 10 == 0 and len(all_preds) > 0:
                running_per = wer_metric.compute(predictions=all_preds, references=all_labels)
                
                # Quick F1 calculation on last 100 samples
                recent_preds = all_preds[-100:]
                recent_labels = all_labels[-100:]
                f1s = []
                for p_str, l_str in zip(recent_preds, recent_labels):
                    p_counts = Counter(p_str.split())
                    l_counts = Counter(l_str.split())
                    tp = sum((p_counts & l_counts).values())
                    fp = sum((p_counts - l_counts).values())
                    fn = sum((l_counts - p_counts).values())
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    f1s.append(f1)
                running_f1 = np.mean(f1s)
                
                pbar.set_postfix({
                    "PER": f"{running_per:.2%}", 
                    "Acc": f"{1-running_per:.2%}",
                    "F1": f"{running_f1:.2%}"
                })
            
    # Report skipped samples
    if skipped > 0:
        warning(f"Skipped {skipped} invalid/corrupted samples")
    
    # Final PER
    per = wer_metric.compute(predictions=all_preds, references=all_labels)
    
    results = {
        "Phoneme Error Rate (PER)": per,
        "Accuracy (1 - PER)": 1.0 - per,
    }
    
    # Bag-of-phonemes F1 (full calculation)
    all_f1s = []
    all_precs = []
    all_recs = []
    
    for p_str, l_str in zip(all_preds, all_labels):
        p_counts = Counter(p_str.split())
        l_counts = Counter(l_str.split())
        
        tp = sum((p_counts & l_counts).values())
        fp = sum((p_counts - l_counts).values())
        fn = sum((l_counts - p_counts).values())
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        all_f1s.append(f1)
        all_precs.append(prec)
        all_recs.append(rec)
        
    results["F1 Score (Bag-of-Phonemes)"] = np.mean(all_f1s)
    results["Precision"] = np.mean(all_precs)
    results["Recall"] = np.mean(all_recs)
    
    return results


def evaluate_model(
    model_path: str,
    data_config: any,
    device: str = None,
    batch_size: int = 8,
    force_reprocess: bool = False,
    val_only: bool = False,
):
    """Evaluate model on validation and test sets, return metrics (PER, Accuracy, F1)."""
    # Setup recognizer
    config = InferenceConfig(model_path=model_path, device=device)
    recognizer = PhonemeRecognizer(config)
    
    # Load dataset
    dataset = prepare_dataset(data_config, recognizer.processor, recognizer.vocab, force_reprocess=force_reprocess, val_only=val_only)
    
    all_results = {}
    
    # Evaluate train set first (if not val_only)
    if not val_only:
        train_set = dataset.get("train")
        if train_set:
            train_results = _evaluate_split(recognizer, train_set, "train", data_config.streaming)
            print_results(train_results, title="Train Set Evaluation")
            all_results["train"] = train_results
        else:
            info("Train set not found in dataset (skipping)")
    
    # Then evaluate validation set
    val_set = dataset.get("validation")
    if val_set:
        val_results = _evaluate_split(recognizer, val_set, "validation", data_config.streaming)
        print_results(val_results, title="Validation Set Evaluation")
        all_results["validation"] = val_results
    else:
        warning("Validation set not found in dataset")
    
    # Finally evaluate test set if it exists
    if not val_only:
        test_set = dataset.get("test")
        if test_set:
            test_results = _evaluate_split(recognizer, test_set, "test", data_config.streaming)
            print_results(test_results, title="Test Set Evaluation")
            all_results["test"] = test_results
        else:
            info("Test set not found in dataset (skipping)")
    
    if not all_results:
        error("No validation or test set found in dataset")
        return None
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        error("Usage: python -m w2v2.inference <audio_file> [model_path]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/phoneme-model"
    
    transcribe_audio(audio_path, model_path)
