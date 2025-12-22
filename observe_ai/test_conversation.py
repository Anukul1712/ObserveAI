"""
test_conversation.py

Loads queries from Config.DATA_DIR/task_2_queries.csv.
Groups rows by 'query_id' to treat them as a continuous conversation (initial + follow-ups).
Sends the list of queries for a specific ID to the /conversation endpoint.
Maps the list of responses back to the rows and appends them to
Config.DATA_DIR/task_2_conversation_responses.csv.
"""

import os
import csv
import json
import requests
from typing import List, Dict, Any
from collections import defaultdict

# Use your project's Config for paths
from src.config import Config
# URL can be overridden with env var API_URL
API_URL = os.environ.get("API_URL", "http://localhost:8000/conversation")

DATASET_PATH = Config.DATA_DIR / "task_2_queries.csv"
SAVE_PATH = Config.DATA_DIR / "task_2_conversation_responses.csv"

HEADERS = {"Content-Type": "application/json"}


def load_rows(path) -> List[Dict[str, Any]]:
    """Reads the CSV into a list of dictionaries."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def append_row_to_csv(path, row: Dict[str, Any], fieldnames: List[str]):
    """Append a single row and ensure it's flushed to disk immediately."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            f.flush()
        writer.writerow(row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def send_conversation_payload(queries: List[str]) -> List[str]:
    """
    Sends a list of queries to the /conversation endpoint.
    Returns a list of string responses corresponding to the queries.
    """
    payload = {"queries": queries}
    resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=10000)
    resp.raise_for_status()
    
    data = resp.json()
    # Expecting: {"results": ["response 1", "response 2"]}
    return data.get("results", [])


def main():
    if not os.path.exists(DATASET_PATH):
        raise SystemExit(f"Dataset not found at {DATASET_PATH}")

    rows = load_rows(DATASET_PATH)
    print(f"Total rows loaded: {len(rows)}")

    # Group rows by query_id to form conversations
    # We use a dict to preserve the order of groups as they appear in the file
    grouped_queries = defaultdict(list)
    for row in rows:
        q_id = row.get("query_id", "unknown")
        grouped_queries[q_id].append(row)

    print(f"Identify {len(grouped_queries)} unique conversation groups.")

    # Output CSV columns
    fieldnames = ["query_id", "query", "response_text", "error", "full_response_json"]

    # Initialize output file if strictly necessary, but append_row handles creation
    if not SAVE_PATH.exists():
         with open(SAVE_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    processed_count = 0

    for query_id, group_rows in grouped_queries.items():
        processed_count += 1
        print(f"[{processed_count}/{len(grouped_queries)}] Processing Group ID: {query_id} ({len(group_rows)} turns)")

        # Extract queries for this group
        # We need to map valid queries to their original rows to handle empty queries gracefully
        valid_indices = []
        queries_to_send = []

        for idx, row in enumerate(group_rows):
            q_text = row.get("query", "").strip()
            if q_text:
                queries_to_send.append(q_text)
                valid_indices.append(idx)
            else:
                # Handle empty query row immediately
                out_row = {
                    "query_id": query_id,
                    "query": "",
                    "response_text": "",
                    "error": "Skipped empty query",
                    "full_response_json": ""
                }
                append_row_to_csv(SAVE_PATH, out_row, fieldnames)

        if not queries_to_send:
            print(f" -> Skipped group {query_id} (No valid queries)")
            continue

        try:
            # Send the conversation list
            responses = send_conversation_payload(queries_to_send)

            # Check if response count matches request count
            if len(responses) != len(queries_to_send):
                print(f" -> Warning: Sent {len(queries_to_send)} queries but got {len(responses)} responses.")

            # Map responses back to the rows
            for i, valid_idx in enumerate(valid_indices):
                original_row = group_rows[valid_idx]
                
                response_text = responses[i] if i < len(responses) else ""
                error_msg = "" if i < len(responses) else "Response missing from API"

                out_row = {
                    "query_id": query_id,
                    "query": original_row.get("query", ""),
                    "response_text": response_text,
                    "error": error_msg,
                    "full_response_json": "" # Optional: populate if you want raw data
                }
                append_row_to_csv(SAVE_PATH, out_row, fieldnames)
            
            print(f" -> Success for group {query_id}")

        except requests.exceptions.RequestException as e:
            err_msg = f"Request failed: {e}"
            print(f" -> ERROR for group {query_id}: {err_msg}")
            
            # Record error for all valid rows in this group
            for valid_idx in valid_indices:
                original_row = group_rows[valid_idx]
                out_row = {
                    "query_id": query_id,
                    "query": original_row.get("query", ""),
                    "response_text": "",
                    "error": err_msg,
                    "full_response_json": ""
                }
                append_row_to_csv(SAVE_PATH, out_row, fieldnames)

    print(f"Done. Responses appended to: {SAVE_PATH}")


if __name__ == "__main__":
    main()