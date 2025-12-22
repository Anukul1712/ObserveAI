import requests
import sys
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

API_URL = "http://localhost:8000"
INTERACTIVE_ENDPOINT = f"{API_URL}/interactive_query"
RESET_ENDPOINT = f"{API_URL}/reset_memory"

console = Console()


def main():
    console.rule(
        "[bold blue]Observe AI Interactive Terminal Client[/bold blue]")
    console.print(
        "Type [bold red]'exit'[/bold red] to stop.", justify="center")
    console.print(
        "Type [bold yellow]'reset'[/bold yellow] to clear conversation memory.", justify="center")
    console.rule()

    # Reset memory on startup
    try:
        with console.status("[bold green]Initializing session...[/bold green]"):
            resp = requests.post(RESET_ENDPOINT)
            resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        console.print(
            f"[bold red]Error clearing memory on startup: {e}[/bold red]")
        return

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]User[/bold cyan]")
            user_input = user_input.strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit"]:
            console.print("[bold red]Exiting...[/bold red]")
            break

        if user_input.lower() in ["reset"]:
            try:
                with console.status("[bold yellow]Resetting memory...[/bold yellow]"):
                    resp = requests.post(RESET_ENDPOINT)
                    resp.raise_for_status()
                console.print(
                    f"[bold green]System: {resp.json().get('message', 'Memory cleared.')}[/bold green]")
            except requests.exceptions.RequestException as e:
                console.print(
                    f"[bold red]Error clearing memory: {e}[/bold red]")
            continue

        # Send query
        try:
            start_time = time.time()

            with console.status("[bold green]Agent is thinking...[/bold green]", spinner="dots"):
                payload = {"query": user_input}
                resp = requests.post(INTERACTIVE_ENDPOINT,
                                     json=payload, timeout=600)
                resp.raise_for_status()

                data = resp.json()
                response_text = data.get("response", "")

            elapsed = time.time() - start_time

            # Format response
            md = Markdown(response_text)
            console.print(
                Panel(md, title="[bold green]Agent[/bold green]", border_style="green"))
            console.print(
                f"[dim](Response time: {elapsed:.2f} seconds)[/dim]", justify="right")

        except requests.exceptions.RequestException as e:
            console.print(
                f"\n[bold red]Error communicating with API: {e}[/bold red]")
        except KeyboardInterrupt:
            console.print(
                "\n[bold red]Operation cancelled by user.[/bold red]")
            break


if __name__ == "__main__":
    main()
