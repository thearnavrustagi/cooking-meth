"""Training script for wav2vec2 phoneme recognition."""

import os

# Enable MPS fallback for unsupported ops (e.g., CTC loss)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import evaluate
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
)

from w2v2.data import (
    DataConfig,
    create_phoneme_vocabulary,
    load_vocabulary,
    prepare_dataset,
    PhonemeDataCollator,
    PAD_TOKEN,
    UNK_TOKEN,
    WORD_DELIMITER_TOKEN,
)
from lib.logger import console, info, success, error, print_config, print_results


@dataclass
class TrainConfig:
    """Configuration for training.
    
    Hyperparameters based on HuggingFace's official wav2vec2/XLS-R fine-tuning guide:
    https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
    
    Key findings:
    - Learning rate: 3e-4 (higher than typical NLP fine-tuning)
    - Dropout: Set ALL dropouts to 0.0 for fine-tuning (model is already regularized)
    - Layerdrop: 0.0 (important for CTC fine-tuning)
    - Freeze feature encoder: Always freeze the CNN feature extractor
    - Gradient checkpointing: Enable for memory efficiency
    - Epochs: 30+ recommended for good convergence
    """
    # Model
    base_model: str = "theainerd/Wav2Vec2-large-xlsr-hindi"
    output_dir: str = "outputs/phoneme-model"
    vocab_path: str = "vocab.json"
    
    # Training hyperparameters (from HuggingFace guide)
    num_train_epochs: int = 30  # Increased from 15 - CTC needs more epochs
    per_device_train_batch_size: int = 4  # Adjust based on GPU memory
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    learning_rate: float = 3e-4  # IMPORTANT: 10x higher than typical (3e-5 was too low!)
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Evaluation & Checkpointing
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 50
    
    # Model config - IMPORTANT: All dropouts should be 0.0 for fine-tuning!
    # The pretrained model already has regularization built in
    attention_dropout: float = 0.0  # Changed from 0.1 - HuggingFace recommends 0.0
    hidden_dropout: float = 0.0     # Changed from 0.1 - HuggingFace recommends 0.0
    feat_proj_dropout: float = 0.0  # Correct
    mask_time_prob: float = 0.05    # SpecAugment-style masking (correct)
    layerdrop: float = 0.0          # Changed from 0.1 - must be 0.0 for CTC fine-tuning!
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    
    # Freezing - Always freeze feature encoder for transfer learning
    freeze_feature_encoder: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = True  # Enable for memory efficiency
    
    # Hardware
    fp16: bool = True   # Enable for CUDA (faster training)
    bf16: bool = False  # Alternative to fp16 for newer GPUs
    use_mps: bool = False  # Disabled - wav2vec2-large too big for MPS memory
    dataloader_num_workers: int = 4
    
    # Training efficiency
    group_by_length: bool = True  # Group similar length samples together


def create_processor(vocab_path: str) -> Wav2Vec2Processor:
    """Create processor with phoneme vocabulary."""
    # Create tokenizer from vocab
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        word_delimiter_token=WORD_DELIMITER_TOKEN,
    )
    
    # Create feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    
    # Combine into processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    
    return processor


def create_model(config: TrainConfig, vocab_size: int) -> Wav2Vec2ForCTC:
    """Create and configure the model for phoneme recognition."""
    info(f"Loading base model: {config.base_model}")
    
    model = Wav2Vec2ForCTC.from_pretrained(
        config.base_model,
        attention_dropout=config.attention_dropout,
        hidden_dropout=config.hidden_dropout,
        feat_proj_dropout=config.feat_proj_dropout,
        mask_time_prob=config.mask_time_prob,
        layerdrop=config.layerdrop,
        ctc_loss_reduction=config.ctc_loss_reduction,
        ctc_zero_infinity=config.ctc_zero_infinity,
        pad_token_id=0,
        vocab_size=vocab_size,
        ignore_mismatched_sizes=True,  # Important: vocab size changes
    )
    
    # Note: Gradient checkpointing is configured via TrainingArguments
    # The trainer will handle enabling it properly
    
    # Freeze feature encoder for stability
    if config.freeze_feature_encoder:
        model.freeze_feature_encoder()
        info("Feature encoder frozen")
    
    # Ensure the classification head is trainable
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    # Log trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return model


def compute_metrics(pred, processor):
    """Compute PER (Phoneme Error Rate) metric."""
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
    
    # Replace -100 with pad token id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    # Compute WER (works as PER since we're working with phonemes)
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"per": wer}


def train(
    train_config: TrainConfig,
    data_config: Optional[DataConfig] = None,
):
    """Run training pipeline."""
    if data_config is None:
        data_config = DataConfig()
    
    # Print configuration
    console.rule("[bold blue]Training Configuration")
    effective_batch = train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps
    print_config({
        "Base Model": train_config.base_model,
        "Output Directory": train_config.output_dir,
        "Epochs": train_config.num_train_epochs,
        "Batch Size": f"{train_config.per_device_train_batch_size} (effective: {effective_batch})",
        "Learning Rate": f"{train_config.learning_rate} (recommended: 3e-4)",
        "Warmup Steps": train_config.warmup_steps,
        "Dropout": f"attn={train_config.attention_dropout}, hidden={train_config.hidden_dropout}",
        "Dataset": f"{data_config.dataset_type} ({data_config.dataset_name})",
        "Language": data_config.language,
        "Max Audio Length": f"{data_config.max_audio_length}s",
    })
    
    # Create output directory
    os.makedirs(train_config.output_dir, exist_ok=True)
    
    # Step 1: Create vocabulary
    console.rule("[bold green]Step 1: Creating Vocabulary")
    vocab_path = os.path.join(train_config.output_dir, "vocab.json")
    vocab = create_phoneme_vocabulary(vocab_path)
    vocab_size = len(vocab)
    success(f"Vocabulary created with {vocab_size} phonemes")
    
    # Step 2: Create processor
    console.rule("[bold green]Step 2: Creating Processor")
    processor = create_processor(vocab_path)
    processor.save_pretrained(train_config.output_dir)
    success("Processor created and saved")
    
    # Step 3: Load and prepare dataset
    console.rule("[bold green]Step 3: Loading Dataset")
    dataset = prepare_dataset(data_config, processor, vocab)
    success(f"Dataset loaded: {len(dataset['train'])} training examples")
    
    # Step 4: Create model
    console.rule("[bold green]Step 4: Initializing Model")
    model = create_model(train_config, vocab_size)
    success("Model initialized")
    
    # Step 5: Setup training
    console.rule("[bold green]Step 5: Setting Up Training")
    
    # Detect device and configure accordingly
    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        use_fp16 = train_config.fp16
        use_bf16 = train_config.bf16
        use_cpu = False
    elif torch.backends.mps.is_available() and train_config.use_mps:
        device_info = "MPS (Apple Silicon) with CPU fallback for CTC loss"
        use_fp16 = False  # MPS doesn't support fp16 training
        use_bf16 = False  # MPS doesn't support bf16 training
        use_cpu = False
    else:
        device_info = "CPU"
        use_fp16 = False
        use_bf16 = False
        use_cpu = True  # Force CPU when MPS disabled or unavailable
    
    info(f"Using device: {device_info}")
    
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        weight_decay=train_config.weight_decay,
        eval_strategy=train_config.eval_strategy,
        save_strategy=train_config.save_strategy,
        save_total_limit=train_config.save_total_limit,
        logging_steps=train_config.logging_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        no_cuda=use_cpu,  # This also disables MPS in newer transformers
        dataloader_num_workers=train_config.dataloader_num_workers,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
        # Memory and efficiency optimizations
        gradient_checkpointing=train_config.gradient_checkpointing if not use_cpu else False,
        group_by_length=train_config.group_by_length,  # Efficient batching by length
    )
    
    data_collator = PhonemeDataCollator(processor=processor, padding=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("test")),
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        processing_class=processor,
    )
    
    # Step 6: Train
    console.rule("[bold green]Step 6: Training")
    info("Starting training...")
    
    train_result = trainer.train()
    
    # Save the final model
    trainer.save_model()
    processor.save_pretrained(train_config.output_dir)
    
    # Print results
    console.rule("[bold blue]Training Complete")
    print_results({
        "Training Loss": train_result.training_loss,
        "Training Steps": train_result.global_step,
        "Model saved to": train_config.output_dir,
    }, title="Training Results")
    
    # Evaluate
    console.rule("[bold green]Evaluation")
    eval_results = trainer.evaluate()
    print_results(eval_results, title="Evaluation Results")
    
    success(f"Training complete! Model saved to {train_config.output_dir}")
    
    return trainer, model, processor


if __name__ == "__main__":
    train_config = TrainConfig()
    data_config = DataConfig()
    train(train_config, data_config)
