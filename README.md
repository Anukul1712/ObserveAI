# Observe AI

Agentic RAG System using LangChain, LangGraph, and Gemini API

## Requirements
- **Python 3.11**
- Google Gemini API Key

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/PriyamPritamPanda/Observe.git
cd Observe
```
Do not forget to be on the right branch

### 2. Create a Python Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r observe_ai/requirements.txt
```

### 4. Create a `.env` File in the Root Folder
Create a file named `.env` in the root directory and add your Gemini API key:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

## Folder Structure
```
Observe/
├── .github/
├── archived/
├── observe_ai/
│   ├── requirements.txt
│   ├── src/
│   │   ├── main.py
│   │   └── ...
├── venv/
├── .env
├── .gitattributes
├── .gitignore
├── README.md
```

## Running the Application
Change directory to `observe_ai` and run the main module:
```bash
cd observe_ai
python -m src.main [ARGS]
```

### Main Arguments
- `--init-vectors` : Initialize vector store from transcripts
- `--interactive`  : Run in interactive mode (CLI)
- `--query "your question"` : Process a single query

#### Examples
Initialize vector store:
```bash
python -m src.main --init-vectors
```

Run in interactive mode:
```bash
python -m src.main --interactive
```

Process a single query:
```bash
python -m src.main --query "What was the outcome of transaction 123?"
```

## Notes
- Ensure your `.env` file contains a valid `GEMINI_API_KEY` before running.
- Python 3.11 is required for compatibility.
- All commands should be run from the `observe_ai` directory.
- src/memory and src/causal are redundant for now. Needs work.
