# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, a local Ollama model for AI generation, and provides a web interface for interaction.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- [Ollama](https://ollama.com) running locally with a tool-capable model: `ollama pull qwen2.5:7b`
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Optional: choose the model**
   
   The default is `qwen2.5:7b` on `http://localhost:11434`. To change it, create a `.env` file in the root directory:
   ```bash
   OLLAMA_MODEL=qwen2.5:7b
   OLLAMA_HOST=http://localhost:11434
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

Windows (PowerShell):

```powershell
winget install Python.Python.3.13
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
uv venv --python "$(py -3.13 -c 'import sys; print(sys.executable)')"
uv sync
cd backend
uv run python -m uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

