"""Rich console logging utilities."""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

# Global console instance
console = Console()

def create_progress() -> Progress:
    """Create a rich progress bar for training/processing."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )

def print_config(config: dict, title: str = "Configuration"):
    """Pretty print a configuration dictionary."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in config.items():
        table.add_row(str(key), str(value))
    
    console.print(table)

def print_results(results: dict, title: str = "Results"):
    """Pretty print results dictionary."""
    table = Table(title=title, show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(str(key), f"{value:.4f}")
        else:
            table.add_row(str(key), str(value))
    
    console.print(table)

def print_phonemes(phonemes: list, timestamps: list = None, title: str = "Phonemes"):
    """Pretty print phoneme list with optional timestamps."""
    table = Table(title=title, show_header=True, header_style="bold blue")
    table.add_column("#", style="dim")
    table.add_column("Phoneme", style="cyan")
    
    if timestamps:
        table.add_column("Start", style="green")
        table.add_column("End", style="green")
        
        for i, (phoneme, (start, end)) in enumerate(zip(phonemes, timestamps)):
            table.add_row(str(i+1), phoneme, f"{start:.3f}s", f"{end:.3f}s")
    else:
        for i, phoneme in enumerate(phonemes):
            table.add_row(str(i+1), phoneme)
    
    console.print(table)

def success(message: str):
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")

def error(message: str):
    """Print an error message."""
    console.print(f"[bold red]✗[/bold red] {message}")

def warning(message: str):
    """Print a warning message."""
    console.print(f"[bold yellow]![/bold yellow] {message}")

def info(message: str):
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")
