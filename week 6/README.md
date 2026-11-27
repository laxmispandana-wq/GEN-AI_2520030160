# Week 6 Generative AI Programming Practical

This folder contains the first baseline evaluation of the Llama 3.2 model against a 30-question golden set. The run is labeled **Release v0.1**. Each response is scored by exact, case-insensitive keyword coverage; a score of 60% or more is marked `PASS`, a lower score is marked `REVIEW`, and a request failure is marked `ERROR`.

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com/download) installed and running
- The `llama3.2` model available in Ollama

## Run

From this folder, install the Python dependency and run the baseline:

```powershell
python -m pip install -r requirements.txt
ollama pull llama3.2
python main.py
```

The `ollama pull` command is only needed if the model is not already installed. Ollama must be serving its local API at `http://localhost:11434` while the script runs.

The script reads `golden_set.json` and writes `baseline_results.csv` beside `main.py`. The CSV includes each model answer, expected answer, matched and missing keywords, score, status, source, model, and release. The score is a simple keyword-coverage measure, not a complete measure of answer quality.
