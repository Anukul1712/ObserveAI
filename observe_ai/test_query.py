import requests
import csv
from pathlib import Path
from typing import List, Any, Dict
from tqdm.auto import tqdm
from src.config import Config


def test_single_query(query: str):
    """
    Sends a single query to the local API and returns the response.
    """
    url = "http://localhost:8000/query"
    payload = {
        "query": query
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        content = data.get("response")
        return content
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return "*****************Failed to get response.**********************"


def process_queries(rows: List[Dict[str, Any]], save_path: Path) -> List[Dict[str, Any]]:
    """
    Iterates through query rows, fetches responses, and writes them to a CSV immediately.
    """
    results = []

    print(f"Starting processing. Saving results to: {save_path}")

    with open(save_path, "w", newline='', encoding='utf-8') as f:
        fieldnames = ["query_id", "query", "response"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, total=len(rows), desc="Processing Queries"):
            query = row.get('query', '')
            query_id = row.get('query_id', 'unknown')

            api_response = test_single_query(query)

            result_row = {
                "query_id": query_id,
                "query": query,
                "response": api_response
            }

            writer.writerow(result_row)
            f.flush()

            results.append(result_row)

    return results


# --- Main Execution ---
if __name__ == "__main__":
    # Define file paths
    dataset_path = Config.DATA_DIR / "task_1_queries.csv"
    save_path = Config.DATA_DIR / "task_1_query_response.csv"

    # Check if input file exists before proceeding
    if not dataset_path.exists():
        print(f"Error: Input file not found at {dataset_path.resolve()}")
        print("Please ensure 'task_1_queries.csv' is in the script directory or update Config.DATA_DIR.")
    else:
        # Load Data
        print(f"Loading data from {dataset_path}...")
        with open(dataset_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"Total rows loaded: {len(rows)}")

        # Run Processing
        if rows:
            all_responses = process_queries(rows, save_path)
            print("Processing complete.")
        else:
            print("The input CSV file is empty.")
