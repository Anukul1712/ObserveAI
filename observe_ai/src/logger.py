import functools
import time
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Define a custom theme for better visual appeal
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "success": "bold green"
})

console = Console(theme=custom_theme)


def setup_logging(level=logging.INFO):
    """
    Configures the logging system to use RichHandler for advanced terminal formatting.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",  # HH:MM:SS
        handlers=[RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_path=False,
            enable_link_path=False,
            markup=True
        )]
    )

    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def log_section(logger, title):
    """
    Helper to log a section header.
    """
    console.rule(f"[bold blue]{title}[/bold blue]")


def log_execution_time(logger=None):
    """
    Decorator to log the start and end of a function execution with time taken.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            func_name = func.__name__
            _logger.info(f"Starting [bold green]{func_name}[/bold green]...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                _logger.info(
                    f"Finished [bold green]{func_name}[/bold green] in [bold yellow]{duration:.2f}s[/bold yellow]")
                return result
            except Exception as e:
                _logger.error(f"Failed [bold red]{func_name}[/bold red]: {e}")
                raise e
        return wrapper
    return decorator
