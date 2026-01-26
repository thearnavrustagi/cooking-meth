"""Data loading and preprocessing for Hindi phoneme recognition."""

import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
import torchaudio
from datasets import load_dataset, Audio, DatasetDict
from transformers import Wav2Vec2Processor
from huggingface_hub import login as hf_login

from w2v2.phoneme import text_to_phoneme_list, VOCABULARY
from lib.logger import console, info, warning, create_progress


def load_hf_token() -> Optional[str]:
    """Load HuggingFace token from environment or .env file."""
    # Try to load from dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    return token


def authenticate_huggingface(token: Optional[str] = None) -> bool:
    """Authenticate with HuggingFace Hub."""
    token = token or load_hf_token()
    
    if token:
        try:
            hf_login(token=token, add_to_git_credential=False)
            info("Authenticated with HuggingFace Hub")
            return True
        except Exception as e:
            warning(f"Failed to authenticate with HuggingFace: {e}")
            return False
    else:
        warning("No HuggingFace token found. Set HUGGINGFACE_TOKEN in .env file")
        return False


# Special tokens for CTC
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
WORD_DELIMITER_TOKEN = "|"
BLANK_TOKEN = "<blank>"


@dataclass
class DataConfig:
    """Configuration for data loading.
    
    Supported datasets:
    - "fleurs": Google FLEURS Hindi (no auth required, smaller ~2k samples)
    - "indicvoices": AI4Bharat IndicVoices Hindi (requires HuggingFace auth, larger ~7k+ samples)
    
    For indicvoices, set HUGGINGFACE_TOKEN in .env file:
    1. Accept license at https://huggingface.co/datasets/ai4bharat/IndicVoices
    2. Create .env file with: HUGGINGFACE_TOKEN=hf_your_token_here
    """
    # Dataset selection: "fleurs" or "indicvoices"
    dataset_type: str = "indicvoices"  # Default to indicvoices (larger dataset)
    dataset_name: str = "ai4bharat/IndicVoices"  # HuggingFace dataset path
    language: str = "hindi"  # Language code
    sampling_rate: int = 16000
    max_audio_length: float = 15.0  # seconds
    min_audio_length: float = 0.5   # seconds
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    num_proc: int = 4
    cache_dir: Optional[str] = None
    max_train_samples: Optional[int] = None  # Limit training samples (for testing)
    streaming: bool = False  # Use streaming for large datasets


def create_phoneme_vocabulary(output_path: str = "vocab.json") -> dict:
    """Create vocabulary file from phoneme list."""
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, WORD_DELIMITER_TOKEN: 2}
    
    # Add all phonemes from the vocabulary
    for i, phoneme in enumerate(sorted(VOCABULARY)):
        if phoneme not in vocab:
            vocab[phoneme] = len(vocab)
    
    # Save vocabulary
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    info(f"Created vocabulary with {len(vocab)} tokens at {output_path}")
    return vocab


def load_vocabulary(vocab_path: str = "vocab.json") -> tuple[dict, dict]:
    """Load vocabulary from file."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    id_to_phoneme = {v: k for k, v in vocab.items()}
    return vocab, id_to_phoneme


def prepare_dataset(
    config: DataConfig,
    processor: Wav2Vec2Processor,
    vocab: dict,
) -> DatasetDict:
    """Load and prepare dataset for phoneme recognition.
    
    Supports:
    - google/fleurs (hi_in)
    - ai4bharat/IndicVoices (Hindi)
    """
    
    info(f"Loading {config.dataset_name} ({config.language})...")
    
    # Get HF token for authenticated datasets
    hf_token = load_hf_token()
    
    # Load dataset based on type
    if config.dataset_type == "fleurs":
        dataset = load_dataset(
            "google/fleurs",
            "hi_in",
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
        text_column = "transcription"
        audio_column = "audio"
        split_mapping = {"train": "train", "validation": "validation", "test": "test"}
        
    elif config.dataset_type == "indicvoices":
        # IndicVoices dataset - requires HuggingFace authentication
        if hf_token:
            info("Using HuggingFace token from environment")
            authenticate_huggingface(hf_token)
        else:
            warning("No HuggingFace token found. Set HUGGINGFACE_TOKEN in .env file")
            warning("If you get 401 errors, run: huggingface-cli login")
        
        dataset = load_dataset(
            "ai4bharat/IndicVoices",
            "hindi",  # Language config name
            cache_dir=config.cache_dir,
            trust_remote_code=True,
            token=hf_token,  # Pass token for authentication
        )
        text_column = "transcript"
        audio_column = "audio"
        split_mapping = {"train": "train", "valid": "validation", "test": "test"}
        
    else:
        # Custom dataset
        dataset = load_dataset(
            config.dataset_name,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
            token=hf_token,
        )
        text_column = "transcript" if "transcript" in dataset["train"].column_names else "transcription"
        audio_column = "audio"
        split_mapping = {"train": "train", "validation": "validation", "test": "test"}
    
    # Cast audio to target sampling rate
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=config.sampling_rate))
    
    # Limit samples if specified (useful for testing)
    if config.max_train_samples:
        info(f"Limiting to {config.max_train_samples} training samples")
        if "train" in dataset:
            dataset["train"] = dataset["train"].select(range(min(config.max_train_samples, len(dataset["train"]))))
    
    # Get columns to keep
    keep_columns = [audio_column, text_column]
    
    def preprocess_function(batch):
        """Preprocess a batch of examples."""
        audio = batch[audio_column]
        
        # Get audio array
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        
        # Check audio length
        duration = len(audio_array) / sampling_rate
        if duration < config.min_audio_length or duration > config.max_audio_length:
            return {
                "input_values": None,
                "labels": None,
                "valid": False,
            }
        
        # Process audio
        inputs = processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="np",
        )
        
        # Convert text to phonemes
        text = batch[text_column]
        if text is None or not text.strip():
            return {
                "input_values": None,
                "labels": None,
                "valid": False,
            }
        
        try:
            phonemes = text_to_phoneme_list(text)
            
            if not phonemes:  # Empty phoneme list
                return {
                    "input_values": None,
                    "labels": None,
                    "valid": False,
                }
            
            # Convert phonemes to IDs
            labels = []
            for phoneme in phonemes:
                if phoneme in vocab:
                    labels.append(vocab[phoneme])
                else:
                    labels.append(vocab[UNK_TOKEN])
            
        except Exception as e:
            warning(f"Failed to convert text '{text}': {e}")
            return {
                "input_values": None,
                "labels": None,
                "valid": False,
            }
        
        return {
            "input_values": inputs.input_values[0],
            "labels": labels,
            "valid": True,
        }
    
    # Process dataset
    with create_progress() as progress:
        task = progress.add_task("Processing dataset...", total=len(dataset))
        
        processed_dataset = DatasetDict()
        
        for split in dataset.keys():
            info(f"Processing {split} split...")
            
            # Remove columns that aren't needed
            columns_to_remove = [
                col for col in dataset[split].column_names 
                if col not in keep_columns
            ]
            split_data = dataset[split].remove_columns(columns_to_remove)
            
            processed = split_data.map(
                preprocess_function,
                remove_columns=keep_columns,
                num_proc=config.num_proc,
                desc=f"Processing {split}",
            )
            
            # Filter out invalid examples
            processed = processed.filter(lambda x: x["valid"])
            processed = processed.remove_columns(["valid"])
            
            # Map to standard split names
            output_split = split_mapping.get(split, split)
            processed_dataset[output_split] = processed
            progress.advance(task)
    
    train_count = len(processed_dataset.get('train', []))
    val_count = len(processed_dataset.get('validation', []))
    info(f"Dataset prepared: {train_count} train, {val_count} validation examples")
    
    return processed_dataset


class PhonemeDataCollator:
    """Data collator for phoneme recognition training."""
    
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: list[dict]) -> dict:
        # Separate input_values and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [f["labels"] for f in features]
        
        # Pad input_values
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels
        max_label_length = max(len(labels) for labels in label_features)
        
        padded_labels = []
        for labels in label_features:
            padding_length = max_label_length - len(labels)
            padded_labels.append(labels + [-100] * padding_length)
        
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch


def get_sample_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load and resample audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze()
