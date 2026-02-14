"""Training script for wav2vec2 phoneme recognition."""

import os

# Enable MPS fallback for unsupported ops (e.g., CTC loss)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_callback import ProgressCallback
from tqdm import tqdm

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
from lib.logger import console, info, success, error, warning, print_config, print_results


class CustomProgressCallback(ProgressCallback):
    """Custom progress callback that shows more metrics on the progress bar."""
    
    def __init__(self):
        super().__init__()
        self.last_eval_metrics = {}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update progress bar with current metrics."""
        if state.is_world_process_zero and self.training_bar is not None:
            # Build postfix dict for progress bar
            postfix = {}
            
            if logs:
                # Training metrics
                if "loss" in logs:
                    postfix["loss"] = f"{logs['loss']:.3f}"
                if "learning_rate" in logs:
                    postfix["lr"] = f"{logs['learning_rate']:.1e}"
                if "epoch" in logs:
                    postfix["epoch"] = f"{logs['epoch']:.2f}"
            
            # Add last evaluation metrics if available
            if self.last_eval_metrics:
                postfix["acc"] = f"{self.last_eval_metrics.get('acc', 0)*100:.1f}%"
                postfix["f1"] = f"{self.last_eval_metrics.get('f1', 0)*100:.1f}%"
            
            self.training_bar.set_postfix(postfix)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Store evaluation metrics for display on progress bar."""
        if metrics:
            self.last_eval_metrics = {
                "per": metrics.get("eval_per", 0),
                "acc": metrics.get("eval_accuracy", 0),
                "f1": metrics.get("eval_f1", 0),
            }


class MetricsLoggingCallback(TrainerCallback):
    """Callback to display detailed metrics after each evaluation."""
    
    def __init__(self):
        self.progress_callback = None  # Will be set by trainer setup
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation - print detailed summary and update progress bar."""
        if metrics is None:
            return
        
        eval_loss = metrics.get("eval_loss", 0)
        per = metrics.get("eval_per", 0)
        accuracy = metrics.get("eval_accuracy", 0)
        f1 = metrics.get("eval_f1", 0)
        precision = metrics.get("eval_precision", 0)
        recall = metrics.get("eval_recall", 0)
        
        # Update progress callback's metrics so they show on progress bar
        if self.progress_callback is not None:
            self.progress_callback.last_eval_metrics = {
                "per": per,
                "acc": accuracy,
                "f1": f1,
            }
        
        # Print detailed summary
        console.print()
        console.print(f"[bold cyan]{'═'*50}[/bold cyan]")
        console.print(f"[bold cyan]  EPOCH {state.epoch:.0f} EVALUATION RESULTS[/bold cyan]")
        console.print(f"[bold cyan]{'═'*50}[/bold cyan]")
        console.print(f"  [bold]Loss:[/bold]      {eval_loss:.4f}")
        console.print(f"  [bold]PER:[/bold]       {per*100:.1f}%")
        
        # Color coding
        acc_color = "green" if accuracy >= 0.7 else ("yellow" if accuracy >= 0.4 else "red")
        f1_color = "green" if f1 >= 0.7 else ("yellow" if f1 >= 0.4 else "red")
        
        console.print(f"  [bold]Accuracy:[/bold]  [{acc_color}]{accuracy*100:.1f}%[/{acc_color}]")
        console.print(f"  [bold]F1 Score:[/bold]  [{f1_color}]{f1*100:.1f}%[/{f1_color}]")
        console.print(f"  [bold]Precision:[/bold] {precision*100:.1f}%")
        console.print(f"  [bold]Recall:[/bold]    {recall*100:.1f}%")
        console.print(f"[bold cyan]{'═'*50}[/bold cyan]")
        console.print()


class SustainedPredictionTrainer(Trainer):
    """Custom trainer that adds regularization for sustained phoneme predictions.
    
    Standard CTC training often produces sparse predictions where each phoneme
    is predicted for only 1 frame, then blanks, then another phoneme. This results
    in very short (20ms) phoneme durations that don't match natural speech.
    
    This trainer adds two regularization terms:
    1. Temporal smoothness: Penalizes rapid changes between consecutive frames
    2. Blank penalty: Discourages excessive blank predictions
    """
    
    def __init__(self, *args, temporal_smoothness: float = 0.0, blank_penalty: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_smoothness = temporal_smoothness
        self.blank_penalty = blank_penalty
        self.blank_token_id = 0  # <pad> is typically the blank token in CTC
        self._logged_reg = False
        
        if temporal_smoothness > 0 or blank_penalty > 0:
            info(f"Sustained prediction regularization enabled:")
            info(f"  - Temporal smoothness: {temporal_smoothness}")
            info(f"  - Blank penalty: {blank_penalty}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with additional regularization terms."""
        # Get model outputs
        outputs = model(**inputs)
        
        # Standard CTC loss from the model
        loss = outputs.loss
        
        if self.temporal_smoothness > 0 or self.blank_penalty > 0:
            logits = outputs.logits  # [batch, time, vocab]
            
            # Use log_softmax for numerical stability, then exp only where needed
            # This is more memory efficient than computing full softmax
            
            # 1. Temporal smoothness loss
            # Penalize differences between consecutive frame logits (before softmax)
            # This is faster and uses less memory than computing full probability distributions
            if self.temporal_smoothness > 0:
                # Use logits directly - penalize changes in logit space
                temporal_diff = logits[:, 1:, :] - logits[:, :-1, :]
                smoothness_loss = torch.mean(temporal_diff ** 2) * 0.001  # Scale down since logits are larger
                loss = loss + self.temporal_smoothness * smoothness_loss
            
            # 2. Blank penalty loss
            # Penalize high probability of blank token
            if self.blank_penalty > 0:
                # Only compute softmax for blank token column (much faster)
                # Use logsumexp trick: softmax[i] = exp(x[i]) / sum(exp(x))
                # blank_prob = exp(logits[:,:,0]) / exp(logsumexp(logits, dim=-1))
                log_probs = F.log_softmax(logits, dim=-1)
                blank_log_probs = log_probs[:, :, self.blank_token_id]
                blank_loss = torch.mean(torch.exp(blank_log_probs))
                loss = loss + self.blank_penalty * blank_loss
            
            # Log once to confirm regularization is being applied
            if not self._logged_reg:
                self._logged_reg = True
                print(f"[Regularization] smoothness_loss={smoothness_loss.item():.4f}, blank_loss={blank_loss.item():.4f}" if self.temporal_smoothness > 0 and self.blank_penalty > 0 else "")
        
        return (loss, outputs) if return_outputs else loss


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
    
    Regularization for sustained predictions:
    - temporal_smoothness: Penalizes rapid changes between consecutive frames (0.1-0.5 recommended)
    - blank_penalty: Discourages excessive blank predictions (0.1-0.3 recommended)
    """
    # Model
    base_model: str = "theainerd/Wav2Vec2-large-xlsr-hindi"
    output_dir: str = "outputs/phoneme-model-sustained"  # New default to avoid overwriting
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
    eval_strategy: str = "epoch"  # Evaluate after every epoch
    eval_steps: int = 500  # Only used if eval_strategy="steps"
    save_strategy: str = "epoch"  # Save after every epoch
    save_steps: int = 500  # Only used if save_strategy="steps"
    save_total_limit: int = 3  # Keep last 3 checkpoints
    
    # Logging
    logging_steps: int = 25  # Log more frequently for better visibility
    
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
    group_by_length: bool = False  # Disabled - can cause slow startup with large datasets
    
    # === Regularization for sustained predictions ===
    # These help the model produce longer, more stable phoneme predictions
    # instead of sparse 1-frame predictions with blanks in between.
    
    # Temporal smoothness: penalizes differences between consecutive frame predictions
    # Higher values = smoother predictions (frames more similar to neighbors)
    # Recommended: 0.1-0.5 (start with 0.2)
    temporal_smoothness: float = 0.2
    
    # Blank penalty: discourages excessive blank token predictions
    # Higher values = fewer blanks, more phoneme predictions
    # Recommended: 0.1-0.3 (start with 0.1)
    blank_penalty: float = 0.1


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation from 20ms to 5ms model.
    
    The teacher model (20ms) has CNN strides [5, 2, 2, 2, 2, 2, 2] = 320 total stride.
    The student model (5ms) has CNN strides [5, 2, 2, 2, 1, 1, 1] = 80 total stride (4x finer).
    
    Distillation trains the student to match the teacher's outputs by:
    1. Running both models on the same input
    2. Upsampling teacher outputs 4x to match student frame rate
    3. Using KL divergence loss between teacher and student logits
    4. Optionally combining with CTC loss on labels
    """
    # Teacher model (existing 20ms model)
    teacher_model_path: str = "outputs/phoneme-model"
    
    # Student model output (new 5ms model)
    student_output_dir: str = "outputs/phoneme-model-5ms"
    
    # Training hyperparameters
    num_train_epochs: int = 15
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4  # Lower than from-scratch training
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Distillation parameters
    temperature: float = 2.0  # Softmax temperature for distillation
    alpha: float = 0.7  # Weight for distillation loss (1-alpha for CTC loss)
    
    # Evaluation & Checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 50
    
    # Student model CNN config for 5ms frames
    # Default wav2vec2: conv_kernel=[10,3,3,3,3,2,2], conv_stride=[5,2,2,2,2,2,2] = 320 stride
    # 5ms model: conv_kernel=[10,3,3,3,3,2,2], conv_stride=[5,2,2,2,1,1,1] = 80 stride
    student_conv_kernel: tuple = (10, 3, 3, 3, 3, 2, 2)
    student_conv_stride: tuple = (5, 2, 2, 2, 1, 1, 1)  # Total stride = 80 (5ms at 16kHz)
    
    # Hardware
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 4


def create_student_model_config(
    teacher_config: Wav2Vec2Config,
    distill_config: DistillationConfig,
    vocab_size: int,
) -> Wav2Vec2Config:
    """Create a student model config with modified CNN for 5ms frame rate."""
    # Start with teacher's config
    student_config = copy.deepcopy(teacher_config)
    
    # Modify CNN configuration for 5ms frames
    student_config.conv_kernel = list(distill_config.student_conv_kernel)
    student_config.conv_stride = list(distill_config.student_conv_stride)
    
    # Update vocab size
    student_config.vocab_size = vocab_size
    
    # Calculate and log frame rate
    total_stride = 1
    for s in distill_config.student_conv_stride:
        total_stride *= s
    frame_ms = total_stride / 16  # 16 samples per ms at 16kHz
    
    info(f"Student model CNN stride: {distill_config.student_conv_stride}")
    info(f"Total stride: {total_stride} samples = {frame_ms}ms per frame")
    
    return student_config


def create_student_model(
    teacher_model: Wav2Vec2ForCTC,
    distill_config: DistillationConfig,
    vocab_size: int,
) -> Wav2Vec2ForCTC:
    """Create a student model with 5ms frame rate, initialized from teacher where possible."""
    info("Creating student model with 5ms frame rate...")
    
    # Create student config
    student_config = create_student_model_config(
        teacher_model.config,
        distill_config,
        vocab_size,
    )
    
    # Create student model with new config
    student_model = Wav2Vec2ForCTC(student_config)
    
    # Copy weights from teacher where dimensions match
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    
    copied_layers = 0
    skipped_layers = 0
    
    for name, param in teacher_state.items():
        if name in student_state:
            if student_state[name].shape == param.shape:
                student_state[name] = param.clone()
                copied_layers += 1
            else:
                skipped_layers += 1
                info(f"  Shape mismatch for {name}: teacher={param.shape}, student={student_state[name].shape}")
    
    student_model.load_state_dict(student_state)
    
    info(f"Copied {copied_layers} layers from teacher, skipped {skipped_layers} due to shape mismatch")
    
    # Log trainable parameters
    trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student_model.parameters())
    info(f"Student trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return student_model


class DistillationTrainer:
    """Trainer for knowledge distillation from teacher (20ms) to student (5ms) model."""
    
    def __init__(
        self,
        teacher_model: Wav2Vec2ForCTC,
        student_model: Wav2Vec2ForCTC,
        config: DistillationConfig,
        processor: Wav2Vec2Processor,
        train_dataset,
        eval_dataset,
        device: str = "cuda",
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = torch.device(device)
        
        # Move models to device
        self.teacher.to(self.device)
        self.student.to(self.device)
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup data collator
        self.data_collator = PhonemeDataCollator(processor=processor, padding=True)
        
        # Calculate frame ratio (teacher to student)
        teacher_stride = 1
        for s in teacher_model.config.conv_stride:
            teacher_stride *= s
        student_stride = 1
        for s in student_model.config.conv_stride:
            student_stride *= s
        self.frame_ratio = teacher_stride // student_stride
        info(f"Frame ratio (teacher/student): {self.frame_ratio}x")
    
    def compute_distillation_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Compute KL divergence loss between teacher and student logits.
        
        The teacher outputs fewer frames (20ms) than student (5ms), so we need to
        upsample teacher frames to match student frame count.
        """
        # Upsample teacher logits to match student frame count
        # teacher_logits: [batch, teacher_frames, vocab]
        # student_logits: [batch, student_frames, vocab]
        
        teacher_frames = teacher_logits.shape[1]
        student_frames = student_logits.shape[1]
        
        # Interpolate teacher logits to student frame count
        # Reshape for interpolation: [batch, vocab, frames]
        teacher_logits_t = teacher_logits.transpose(1, 2)
        teacher_upsampled = F.interpolate(
            teacher_logits_t,
            size=student_frames,
            mode='linear',
            align_corners=True,
        )
        teacher_upsampled = teacher_upsampled.transpose(1, 2)  # Back to [batch, frames, vocab]
        
        # Compute soft targets with temperature
        teacher_probs = F.softmax(teacher_upsampled / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL divergence loss (scaled by T^2 as per Hinton et al.)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
        ) * (temperature ** 2)
        
        return kl_loss
    
    def compute_ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss for student model."""
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        
        # Get target lengths (non -100 values)
        target_lengths = (labels != -100).sum(dim=1)
        
        # Replace -100 with 0 for CTC loss
        labels_for_ctc = labels.clone()
        labels_for_ctc[labels_for_ctc == -100] = 0
        
        ctc_loss = F.ctc_loss(
            log_probs,
            labels_for_ctc,
            input_lengths,
            target_lengths,
            blank=self.processor.tokenizer.pad_token_id,
            reduction='mean',
            zero_infinity=True,
        )
        
        return ctc_loss
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.student.train()
        
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.config.dataloader_num_workers,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            input_values = batch["input_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Get teacher outputs (no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher(input_values)
                teacher_logits = teacher_outputs.logits
            
            # Get student outputs
            student_outputs = self.student(input_values)
            student_logits = student_outputs.logits
            
            # Compute distillation loss
            distill_loss = self.compute_distillation_loss(
                teacher_logits,
                student_logits,
                self.config.temperature,
            )
            
            # Compute CTC loss on student
            input_lengths = torch.full(
                (input_values.shape[0],),
                student_logits.shape[1],
                dtype=torch.long,
                device=self.device,
            )
            ctc_loss = self.compute_ctc_loss(student_logits, labels, input_lengths)
            
            # Combined loss
            alpha = self.config.alpha
            loss = alpha * distill_loss + (1 - alpha) * ctc_loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += accumulated_loss
                num_batches += 1
                accumulated_loss = 0.0
            
            pbar.set_postfix({
                "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                "distill": f"{distill_loss.item():.4f}",
                "ctc": f"{ctc_loss.item():.4f}",
            })
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self) -> dict:
        """Evaluate on validation set."""
        self.student.eval()
        
        dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.config.dataloader_num_workers,
        )
        
        wer_metric = evaluate.load("wer")
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.student(input_values)
                pred_ids = torch.argmax(outputs.logits, dim=-1)
                
                # Decode predictions
                pred_str = self.processor.batch_decode(pred_ids)
                
                # Decode labels (replace -100 with pad token)
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                label_str = self.processor.batch_decode(labels, group_tokens=False)
                
                all_preds.extend(pred_str)
                all_labels.extend(label_str)
        
        per = wer_metric.compute(predictions=all_preds, references=all_labels)
        
        return {"per": per, "accuracy": 1.0 - per}
    
    def save_model(self, output_dir: str):
        """Save the student model."""
        os.makedirs(output_dir, exist_ok=True)
        self.student.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        success(f"Student model saved to {output_dir}")
    
    def train(self):
        """Full training loop."""
        console.rule("[bold green]Starting Knowledge Distillation Training")
        
        best_per = float('inf')
        
        for epoch in range(self.config.num_train_epochs):
            console.rule(f"[bold blue]Epoch {epoch + 1}/{self.config.num_train_epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            info(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            metrics = self.evaluate()
            info(f"Validation PER: {metrics['per']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
            # Save best model
            if metrics['per'] < best_per:
                best_per = metrics['per']
                self.save_model(self.config.student_output_dir)
                success(f"New best model saved (PER: {best_per:.4f})")
        
        console.rule("[bold green]Distillation Complete")
        print_results({
            "Best PER": best_per,
            "Best Accuracy": 1.0 - best_per,
            "Model saved to": self.config.student_output_dir,
        }, title="Distillation Results")


def distill(
    distill_config: DistillationConfig,
    data_config: Optional[DataConfig] = None,
    force_reprocess: bool = False,
):
    """Run knowledge distillation to create a 5ms model from a 20ms teacher."""
    if data_config is None:
        data_config = DataConfig()
    
    # Print configuration
    console.rule("[bold blue]Knowledge Distillation Configuration")
    print_config({
        "Teacher Model": distill_config.teacher_model_path,
        "Student Output": distill_config.student_output_dir,
        "Epochs": distill_config.num_train_epochs,
        "Batch Size": f"{distill_config.per_device_train_batch_size} (effective: {distill_config.per_device_train_batch_size * distill_config.gradient_accumulation_steps})",
        "Learning Rate": distill_config.learning_rate,
        "Temperature": distill_config.temperature,
        "Alpha (distill weight)": distill_config.alpha,
        "Teacher Frame Rate": "20ms (stride 320)",
        "Student Frame Rate": "5ms (stride 80)",
    })
    
    # Verify teacher model exists
    if not os.path.exists(distill_config.teacher_model_path):
        error(f"Teacher model not found at {distill_config.teacher_model_path}")
        return None
    
    # Verify student output directory doesn't overwrite teacher
    if os.path.abspath(distill_config.teacher_model_path) == os.path.abspath(distill_config.student_output_dir):
        error("Student output directory cannot be the same as teacher model path!")
        return None
    
    # Create student output directory
    os.makedirs(distill_config.student_output_dir, exist_ok=True)
    
    # Step 1: Load teacher model and processor
    console.rule("[bold green]Step 1: Loading Teacher Model")
    info(f"Loading teacher from {distill_config.teacher_model_path}...")
    
    teacher_model = Wav2Vec2ForCTC.from_pretrained(distill_config.teacher_model_path)
    processor = Wav2Vec2Processor.from_pretrained(distill_config.teacher_model_path)
    
    # Load vocabulary
    vocab_path = os.path.join(distill_config.teacher_model_path, "vocab.json")
    vocab, _ = load_vocabulary(vocab_path)
    vocab_size = len(vocab)
    
    success(f"Teacher model loaded (vocab size: {vocab_size})")
    
    # Step 2: Create student model
    console.rule("[bold green]Step 2: Creating Student Model (5ms)")
    student_model = create_student_model(teacher_model, distill_config, vocab_size)
    success("Student model created")
    
    # Step 3: Load and prepare dataset
    console.rule("[bold green]Step 3: Loading Dataset")
    dataset = prepare_dataset(data_config, processor, vocab, force_reprocess=force_reprocess)
    if data_config.streaming:
        success("Dataset loaded in streaming mode")
    else:
        success(f"Dataset loaded: {len(dataset['train'])} training examples")
    
    # Step 4: Detect device
    console.rule("[bold green]Step 4: Setting Up Training")
    if torch.cuda.is_available():
        device = "cuda"
        info(f"Using CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        info("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        info("Using CPU")
    
    # Step 5: Create trainer and run distillation
    console.rule("[bold green]Step 5: Knowledge Distillation Training")
    
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        config=distill_config,
        processor=processor,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("test")),
        device=device,
    )
    
    trainer.train()
    
    # Save processor and vocab to student output
    processor.save_pretrained(distill_config.student_output_dir)
    
    # Copy vocab file
    import shutil
    shutil.copy(vocab_path, os.path.join(distill_config.student_output_dir, "vocab.json"))
    
    success(f"Distillation complete! 5ms model saved to {distill_config.student_output_dir}")
    
    return trainer


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
    """Compute PER (Phoneme Error Rate), Accuracy, and F1 metrics."""
    from collections import Counter
    
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
    
    # Replace -100 with pad token id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    # Compute WER/PER (works as PER since we're working with phonemes)
    wer_metric = evaluate.load("wer")
    per = wer_metric.compute(predictions=pred_str, references=label_str)
    
    # Compute accuracy (1 - PER)
    accuracy = 1.0 - per
    
    # Compute F1 score (bag-of-phonemes)
    all_f1s = []
    all_precs = []
    all_recs = []
    
    for p_str, l_str in zip(pred_str, label_str):
        # Split into phonemes
        p_phonemes = p_str.split() if p_str else []
        l_phonemes = l_str.split() if l_str else []
        
        p_counts = Counter(p_phonemes)
        l_counts = Counter(l_phonemes)
        
        # Calculate precision, recall, F1
        tp = sum((p_counts & l_counts).values())
        fp = sum((p_counts - l_counts).values())
        fn = sum((l_counts - p_counts).values())
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        all_f1s.append(f1)
        all_precs.append(prec)
        all_recs.append(rec)
    
    avg_f1 = np.mean(all_f1s) if all_f1s else 0
    avg_prec = np.mean(all_precs) if all_precs else 0
    avg_rec = np.mean(all_recs) if all_recs else 0
    
    return {
        "per": per,
        "accuracy": accuracy,
        "f1": avg_f1,
        "precision": avg_prec,
        "recall": avg_rec,
    }


def train(
    train_config: TrainConfig,
    data_config: Optional[DataConfig] = None,
    force_reprocess: bool = False,
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
        "Temporal Smoothness": f"{train_config.temporal_smoothness} (sustained prediction reg.)",
        "Blank Penalty": f"{train_config.blank_penalty} (reduce excessive blanks)",
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
    dataset = prepare_dataset(data_config, processor, vocab, force_reprocess=force_reprocess)
    if data_config.streaming:
        success("Dataset loaded in streaming mode")
    else:
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
    
    # Adjust for streaming
    max_steps = -1
    if data_config.streaming:
        # For IterableDataset, we must set max_steps because we don't know the size
        # We estimate steps based on a heuristic or just set a large number
        # Let's assume ~5000 updates per 'epoch' for a large dataset like IndicVoices
        max_steps = train_config.num_train_epochs * 1000
        info(f"Streaming mode enabled: Setting max_steps={max_steps} (overriding epochs)")

    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        weight_decay=train_config.weight_decay,
        eval_strategy=train_config.eval_strategy,
        eval_steps=train_config.eval_steps,
        save_strategy=train_config.save_strategy,
        save_steps=train_config.save_steps,
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
        group_by_length=train_config.group_by_length if not data_config.streaming else False,  # Efficient batching by length (disabled for streaming)
        # Progress bar settings
        disable_tqdm=False,  # Ensure progress bars are enabled
        logging_first_step=True,  # Log the first step immediately
        # Reduce console spam - our custom callback handles display
        log_level="warning",  # Only show warnings and errors from transformers
    )
    
    data_collator = PhonemeDataCollator(processor=processor, padding=True)
    
    # Create custom callbacks - they share eval metrics
    progress_callback = CustomProgressCallback()
    metrics_callback = MetricsLoggingCallback()
    
    # Link them so progress bar can show eval metrics
    metrics_callback.progress_callback = progress_callback
    
    # Use custom trainer with sustained prediction regularization
    trainer = SustainedPredictionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("test")),
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        processing_class=processor,
        temporal_smoothness=train_config.temporal_smoothness,
        blank_penalty=train_config.blank_penalty,
        callbacks=[metrics_callback],  # Add custom metrics logging
    )
    
    # Replace default progress callback with our custom one
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(progress_callback)
    
    # Step 6: Train
    console.rule("[bold green]Step 6: Training")
    info("Starting training...")
    info("Note: First batch may be slow due to CUDA compilation. Progress bar will appear shortly...")
    
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
