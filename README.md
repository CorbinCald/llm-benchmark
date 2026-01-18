# LLM Benchmark Generator

This tool allows you to benchmark various Large Language Models (LLMs) via the OpenRouter API. It generates code based on a user prompt using multiple models in parallel and saves the results for comparison.

## Prerequisites

- Python 3.8+
- An OpenRouter API key

## Installation

1. Clone the repository (if you haven't already):
   ```bash
   git clone <repository-url>
   cd llm-benchmark
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

You need to provide your OpenRouter API key. You can do this in one of two ways:

1. **Environment Variable**: Set `OPENROUTER_API_KEY` in your environment.
2. **.env File**: Create a `.env` file in the project root containing:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

## Usage

Run the script from the command line:

```bash
python LLM_benchmarks.py
```

You will be prompted to enter a description of the program you want the LLMs to generate.

Alternatively, you can provide the prompt directly as an argument:

```bash
python LLM_benchmarks.py --prompt "Create a snake game in Python"
```

## Output

The script will:
1. Generate a directory name based on your prompt.
2. Create a folder in `benchmarkResults/`.
3. Query multiple models (configured in `MODEL_MAPPING`).
4. Save the generated code from each model into the folder.

## Models

The script is currently configured to test a variety of models including:
- Gemini 3.0 (Flash & Pro)
- GPT-5.2 (and variants)
- Claude (Opus)
- DeepSeek V3.2
- GLM 4.7
- And others...

You can modify the `MODEL_MAPPING` dictionary in `LLM_benchmarks.py` to add or remove models.

