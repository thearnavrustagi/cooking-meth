"""
Gradio web demo for Hindi Phoneme Recognition.

Run with:
    uv run python web_demo.py

Or with a specific model:
    uv run python web_demo.py --model outputs/phoneme-model-5ms
"""

import gradio as gr
import torch
import torchaudio
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import colorsys

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from w2v2.inference import PhonemeRecognizer, InferenceConfig, format_timestamp_srt

# Global recognizer (initialized on startup)
recognizer = None
model_info = {}


def generate_colors(n: int):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7
        lightness = 0.6
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors


def create_waveform_visualization(
    audio_path: str,
    phonemes: list,
    timestamps: list,
    sample_rate: int = 16000,
) -> plt.Figure:
    """
    Create a waveform visualization with phoneme annotations.
    
    Args:
        audio_path: Path to audio file
        phonemes: List of phoneme strings
        timestamps: List of (start, end) tuples in seconds
        sample_rate: Target sample rate
    
    Returns:
        matplotlib Figure object
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    waveform = waveform.squeeze().numpy()
    duration = len(waveform) / sample_rate
    time_axis = np.linspace(0, duration, len(waveform))
    
    # Create figure with two subplots
    fig, (ax_wave, ax_phonemes) = plt.subplots(
        2, 1, 
        figsize=(14, 6),
        height_ratios=[3, 1],
        sharex=True,
        gridspec_kw={'hspace': 0.05}
    )
    
    # Plot waveform
    ax_wave.plot(time_axis, waveform, color='#2196F3', linewidth=0.5, alpha=0.8)
    ax_wave.fill_between(time_axis, waveform, alpha=0.3, color='#2196F3')
    ax_wave.set_ylabel('Amplitude', fontsize=10)
    ax_wave.set_title('Audio Waveform with Phoneme Alignment', fontsize=12, fontweight='bold')
    ax_wave.grid(True, alpha=0.3, axis='x')
    ax_wave.set_xlim(0, duration)
    
    # Get unique phonemes for color mapping
    unique_phonemes = list(set(p for p in phonemes if p != '_'))
    colors = generate_colors(len(unique_phonemes))
    phoneme_colors = {p: colors[i] for i, p in enumerate(unique_phonemes)}
    phoneme_colors['_'] = (0.9, 0.9, 0.9)  # Light gray for blanks
    
    # Add colored regions to waveform
    y_min, y_max = ax_wave.get_ylim()
    for phoneme, (start, end) in zip(phonemes, timestamps):
        color = phoneme_colors.get(phoneme, (0.5, 0.5, 0.5))
        ax_wave.axvspan(start, end, alpha=0.2, color=color)
    
    # Plot phoneme timeline
    ax_phonemes.set_ylim(0, 1)
    ax_phonemes.set_xlim(0, duration)
    ax_phonemes.set_xlabel('Time (seconds)', fontsize=10)
    ax_phonemes.set_yticks([])
    ax_phonemes.set_ylabel('Phonemes', fontsize=10)
    
    # Draw phoneme boxes
    for phoneme, (start, end) in zip(phonemes, timestamps):
        color = phoneme_colors.get(phoneme, (0.5, 0.5, 0.5))
        width = end - start
        
        # Draw rectangle
        rect = Rectangle((start, 0.1), width, 0.8, 
                         facecolor=color, edgecolor='black', 
                         linewidth=0.5, alpha=0.8)
        ax_phonemes.add_patch(rect)
        
        # Add phoneme label if wide enough
        if width > 0.03:  # Only label if segment is wide enough
            # Determine text color based on background brightness
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = 'white' if brightness < 0.5 else 'black'
            
            ax_phonemes.text(
                start + width / 2, 0.5, 
                phoneme,
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color=text_color,
            )
    
    # Add grid lines for time reference
    ax_phonemes.grid(True, alpha=0.3, axis='x')
    
    # Add time markers every 0.5 seconds
    for t in np.arange(0, duration, 0.5):
        ax_wave.axvline(t, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    return fig


def create_phoneme_legend(phonemes: list, timestamps: list) -> plt.Figure:
    """Create a legend showing all phonemes with their colors and durations."""
    # Get unique phonemes with total duration
    phoneme_durations = {}
    for phoneme, (start, end) in zip(phonemes, timestamps):
        if phoneme == '_':
            continue
        if phoneme not in phoneme_durations:
            phoneme_durations[phoneme] = 0
        phoneme_durations[phoneme] += (end - start)
    
    # Sort by duration (most common first)
    sorted_phonemes = sorted(phoneme_durations.items(), key=lambda x: -x[1])
    
    if not sorted_phonemes:
        # Return empty figure
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.text(0.5, 0.5, "No phonemes detected", ha='center', va='center')
        ax.axis('off')
        return fig
    
    # Generate colors
    unique_phonemes = [p for p, _ in sorted_phonemes]
    colors = generate_colors(len(unique_phonemes))
    
    # Create legend figure
    n_cols = min(8, len(sorted_phonemes))
    n_rows = (len(sorted_phonemes) + n_cols - 1) // n_cols
    
    fig, ax = plt.subplots(figsize=(10, max(1, n_rows * 0.5)))
    ax.axis('off')
    
    for i, (phoneme, duration) in enumerate(sorted_phonemes):
        row = i // n_cols
        col = i % n_cols
        
        x = col / n_cols + 0.02
        y = 1 - (row + 0.5) / n_rows
        
        color = colors[unique_phonemes.index(phoneme)]
        
        # Draw color box
        rect = Rectangle((x, y - 0.15), 0.03, 0.3,
                         facecolor=color, edgecolor='black',
                         linewidth=0.5, transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Add label
        ax.text(x + 0.04, y, f"{phoneme} ({duration*1000:.0f}ms)",
               transform=ax.transAxes, fontsize=9, va='center')
    
    ax.set_title("Phoneme Legend (sorted by duration)", fontsize=10, fontweight='bold')
    
    return fig


def init_model(model_path: str = "outputs/phoneme-model"):
    """Initialize the model."""
    global recognizer, model_info
    
    print(f"Initializing model from {model_path}...")
    try:
        config = InferenceConfig(model_path=model_path)
        recognizer = PhonemeRecognizer(config)
        
        # Store model info
        model_info = {
            "path": model_path,
            "frame_duration_ms": config.frame_duration_ms,
            "device": config.device,
            "vocab_size": len(recognizer.vocab),
        }
        
        print(f"Model initialized successfully on {config.device}")
        print(f"Frame duration: {config.frame_duration_ms}ms")
        return True
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        recognizer = None
        return False


def transcribe(audio_path, output_mode: str = "Normal (CTC collapsed)"):
    """Transcribe audio to phonemes."""
    if recognizer is None:
        return "Error: Model not initialized.", "", "", None, None
    
    if audio_path is None:
        return "Please record or upload audio.", "", "", None, None
    
    print(f"Processing audio: {audio_path}")
    
    try:
        # Load audio and get logits
        waveform = recognizer.load_audio(audio_path)
        audio_duration = len(waveform) / recognizer.config.sampling_rate
        logits = recognizer.get_logits(waveform)
        
        # Choose decoding mode
        raw_mode = "Raw" in output_mode
        
        if raw_mode:
            phonemes, timestamps = recognizer.greedy_decode_raw(logits)
        else:
            phonemes, timestamps = recognizer.greedy_decode(logits)
            # Extend timestamps to fill gaps (more useful for visualization)
            phonemes, timestamps = recognizer.extend_to_fill_gaps(
                phonemes, timestamps, audio_duration
            )
        
        # Format as SRT
        srt_output = ""
        for idx, (phoneme, (start, end)) in enumerate(zip(phonemes, timestamps), 1):
            if end - start < 0.01:
                end = start + 0.01
            srt_output += f"{idx}\n"
            srt_output += f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n"
            srt_output += f"{phoneme}\n\n"
        
        # Create phoneme sequence (collapsed, for easy reading)
        if raw_mode:
            # For raw mode, show a summary
            non_blank = [p for p in phonemes if p != "_"]
            phoneme_seq = " ".join(non_blank)
            phoneme_display = f"Collapsed: {phoneme_seq}\n\nRaw sequence ({len(phonemes)} spans):\n"
            # Show first 50 spans
            raw_spans = [f"{p}({(end-start)*1000:.0f}ms)" for p, (start, end) in zip(phonemes[:50], timestamps[:50])]
            phoneme_display += " ".join(raw_spans)
            if len(phonemes) > 50:
                phoneme_display += f"\n... and {len(phonemes) - 50} more spans"
        else:
            phoneme_seq = " ".join(phonemes)
            phoneme_display = phoneme_seq
        
        # Statistics
        pred_ids = np.argmax(logits, axis=-1)
        unique, counts = np.unique(pred_ids, return_counts=True)
        
        blank_count = 0
        for u, c in zip(unique, counts):
            if u == recognizer.config.blank_token_id:
                blank_count = c
                break
        
        total_frames = len(pred_ids)
        blank_pct = (blank_count / total_frames) * 100 if total_frames > 0 else 0
        
        stats = f"""**Audio Duration:** {audio_duration:.2f}s
**Total Frames:** {total_frames}
**Frame Duration:** {recognizer.config.frame_duration_ms}ms
**Blank Frames:** {blank_count} ({blank_pct:.1f}%)
**Phonemes Detected:** {len(phonemes) if not raw_mode else len([p for p in phonemes if p != '_'])}
**Model:** {model_info.get('path', 'unknown')}
**Device:** {model_info.get('device', 'unknown')}"""
        
        # Create visualizations
        try:
            waveform_fig = create_waveform_visualization(
                audio_path, phonemes, timestamps, 
                sample_rate=recognizer.config.sampling_rate
            )
            legend_fig = create_phoneme_legend(phonemes, timestamps)
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
            waveform_fig = None
            legend_fig = None
        
        return srt_output, phoneme_display, stats, waveform_fig, legend_fig
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error during processing: {str(e)}", "", "", None, None


def change_model(model_path):
    """Change the active model."""
    success = init_model(model_path)
    if success:
        return f"Model changed to: {model_path}"
    else:
        return f"Failed to load model: {model_path}"


def create_demo():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Hindi Phoneme Recognizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎤 Hindi Phoneme Recognizer
        
        Record or upload audio to get time-aligned phoneme transcriptions with waveform visualization.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                available_models = []
                for path in ["outputs/phoneme-model", "outputs/phoneme-model-sustained", "outputs/phoneme-model-5ms"]:
                    if os.path.exists(path):
                        available_models.append(path)
                
                if not available_models:
                    available_models = ["outputs/phoneme-model"] # Fallback
                
                model_selector = gr.Dropdown(
                    choices=available_models,
                    value=model_info.get("path", available_models[0]),
                    label="Select Model",
                    info="Choose the model to use for transcription."
                )
                
                model_status = gr.Markdown(value=f"Active model: {model_info.get('path', 'None')}")
                
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Input Audio",
                )
                
                output_mode = gr.Radio(
                    choices=["Normal (CTC collapsed)", "Raw (exact model output)"],
                    value="Normal (CTC collapsed)",
                    label="Output Mode",
                    info="Normal: Merges repeated phonemes. Raw: Shows every frame including blanks."
                )
                
                transcribe_btn = gr.Button("🎯 Transcribe", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                stats_output = gr.Markdown(label="Statistics", value="*Upload audio to see statistics*")
        
        # Waveform visualization
        gr.Markdown("### 📊 Waveform with Phoneme Alignment")
        waveform_plot = gr.Plot(label="Waveform Visualization")
        
        # Legend
        legend_plot = gr.Plot(label="Phoneme Legend")
        
        with gr.Row():
            with gr.Column(scale=1):
                phoneme_output = gr.Textbox(
                    label="Phoneme Sequence",
                    lines=4,
                    placeholder="Phonemes will appear here...",
                )
            
            with gr.Column(scale=1):
                srt_output = gr.TextArea(
                    label="SRT Output (with timestamps)",
                    lines=15,
                    placeholder="SRT content will appear here...",
                )
        
        # Connect the button
        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, output_mode],
            outputs=[srt_output, phoneme_output, stats_output, waveform_plot, legend_plot],
        )
        
        # Model change event
        model_selector.change(
            fn=change_model,
            inputs=[model_selector],
            outputs=[model_status],
        )
        
        # Also transcribe when audio changes
        audio_input.change(
            fn=transcribe,
            inputs=[audio_input, output_mode],
            outputs=[srt_output, phoneme_output, stats_output, waveform_plot, legend_plot],
        )
        
        # Re-transcribe when mode changes (if audio already loaded)
        output_mode.change(
            fn=transcribe,
            inputs=[audio_input, output_mode],
            outputs=[srt_output, phoneme_output, stats_output, waveform_plot, legend_plot],
        )
        
        gr.Markdown("""
        ---
        ### 🎨 Visualization Guide
        
        - **Waveform (top):** Shows the audio amplitude over time with colored regions for each phoneme
        - **Phoneme Timeline (bottom):** Shows each phoneme as a colored box with its label
        - **Legend:** Shows all detected phonemes sorted by total duration
        
        ### Output Modes
        
        - **Normal (CTC collapsed):** Standard output with repeated phonemes merged. 
          Blanks are ignored unless there's a long silence (>200ms).
        - **Raw (exact model output):** Shows exactly what the model predicts for each frame.
          `_` represents blank/pad tokens (shown in gray). Useful for debugging.
        
        ### Tips
        
        - Speak clearly and at a normal pace
        - Avoid background noise for best results
        - The model is trained on Hindi speech
        - Use Raw mode to see if the model is outputting many blanks (gray regions)
        """)
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio web demo for phoneme recognition")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="outputs/phoneme-model",
        help="Path to the trained model (default: outputs/phoneme-model)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    args = parser.parse_args()
    
    # Initialize model
    if not init_model(args.model):
        print("Warning: Model failed to initialize. The demo will show errors.")
    
    # Create and launch demo
    demo = create_demo()
    print(f"\nStarting Gradio server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
