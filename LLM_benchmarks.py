#!/usr/bin/env python3
"""
LLM Benchmark Generator

This script allows benchmarking multiple Large Language Models (LLMs) via the OpenRouter API.
It takes a user prompt, sends it to configured models, and saves the generated code
into organized directories for comparison.
"""

import os
import json
import re
import asyncio
import aiohttp
import argparse

# Configuration
# Try to load from .env file manually
env_key = os.environ.get("OPENROUTER_API_KEY")
if not env_key and os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "OPENROUTER_API_KEY":
                        env_key = value.strip().strip('"').strip("'")
                        break
    except Exception as e:
        print(f"Warning: Could not read .env file: {e}")

API_KEY = env_key
API_URL = "https://openrouter.ai/api/v1"
SYSTEM_PROMPT = (
    "You are an expert programmer. Your goal is to provide a complete, "
    "fully functional, single-file implementation based on the user's request. "
    "Do not include any external modules or dependencies. "
    "Return ONLY the code, with no preamble or explanation."
)
OUTPUT_DIR = "benchmarkResults"

if not API_KEY:
    print("Please set the OPENROUTER_API_KEY environment variable.")
    exit(1)

# Manual mapping of model names to OpenRouter model IDs
MODEL_MAPPING = {
    'claudeOpus4.5': 'anthropic/claude-opus-4.5',
    'gemini3_0Flash': 'google/gemini-3-flash-preview',
    'gemini3_0Pro': 'google/gemini-3-pro-preview',
    #'gpt5_2': 'openai/gpt-5.2',
    #'deepseekV3_2': 'deepseek/deepseek-v3.2',
    'mimoV2Flash': 'xiaomi/mimo-v2-flash',
    'glm4_7': 'z-ai/glm-4.7',
    'gpt5_2Codex': 'openai/gpt-5.2-codex',
    'kimik2_5': 'moonshotai/kimi-k2.5'
}

def get_target_models():
    """
    Returns the list of (model_name, model_id) pairs to process.
    """
    targets = []
    
    # Process all entries in MODEL_MAPPING directly
    for model_name, model_id in MODEL_MAPPING.items():
        targets.append((model_name, model_id))
    
    return targets

async def get_directory_name(session, prompt):
    """
    Uses Gemini to generate a directory name based on the prompt.
    """
    print("Generating directory name...")
    model = 'google/gemini-2.5-flash'
    naming_prompt = f"Generate a short, concise, snake_case directory name (max 3 words) that summarizes the following prompt. Return ONLY the directory name, no other text, no markdown formatting.\n\nPrompt: {prompt}"
    
    try:
        # We reuse the existing call logic but skip extract_code since we want the raw response
        name = await call_model_async(session, model, naming_prompt)
        if name:
            clean_name = name.strip().replace('`', '').strip()
            # Basic validation to ensure it's a valid directory name
            clean_name = "".join(x for x in clean_name if (x.isalnum() or x in "._- "))
            return clean_name
    except Exception as e:
        print(f"Error generating directory name: {e}")
    
    return "benchmark_output"

def get_unique_filename(directory, base_name, extension):
    """
    Generates a unique filename with versioning if needed.
    """
    # Remove 'Cursor' (case insensitive) from base name
    base = re.sub(r'cursor', '', base_name, flags=re.IGNORECASE)
    
    if not extension.startswith('.'):
        extension = f".{extension}"
    
    # Check if file exists
    full_path = os.path.join(directory, f"{base}{extension}")
    if not os.path.exists(full_path):
        return f"{base}{extension}"
        
    counter = 2
    while True:
        new_name = f"{base}_v{counter}{extension}"
        if not os.path.exists(os.path.join(directory, new_name)):
            return new_name
        counter += 1

async def parse_with_gemini(session, content):
    """
    Uses Gemini to parse the content, extract code, and determine the filename.
    """
    print("Parsing content with Gemini...")
    model = 'google/gemini-2.5-flash'
    parsing_prompt = (
        "You are a code extraction engine. Your task is to extract the executable code "
        "from the provided text. \n"
        "1. Identify the programming language.\n"
        "2. Extract the FULL code content without any markdown formatting.\n"
        "3. Determine the correct file extension (e.g., .html, .py, .js).\n"
        "Return a raw JSON object with keys: 'code', 'extension', 'language'. "
        "Do not include any markdown formatting or explanation.\n\n"
        f"Text to parse:\n{content[:50000]}"
    )

    try:
        # Pass reasoning_effort="minimal" for minimal reasoning during normalization
        response = await call_model_async(session, model, parsing_prompt, reasoning_effort="minimal")
        
        if not response:
            return None
            
        # Clean response (remove ```json ... ```)
        clean_resp = response.strip()
        if "```" in clean_resp:
            # simple extraction of json block
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_resp, re.DOTALL)
            if match:
                clean_resp = match.group(1)
        
        return json.loads(clean_resp)
    except Exception as e:
        print(f"Error parsing with Gemini: {e}")
        return None

async def call_model_async(session, model_id, prompt, reasoning_effort="high"):
    """
    Calls the OpenRouter API for a specific model asynchronously.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json", 
        "X-Title": "Benchmark Script"
    }
    
    # Base payload parameters
    base_data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 120000,
    }

    # Set temperature to 1 for Gemini 3 models
    if "gemini-3" in model_id.lower():
        base_data["temperature"] = 1.0

    # Attempt 1: Try with specified reasoning (unless disabled for specific models)
    # Disabled for mimo and glm-4.7
    if reasoning_effort and "mimo" not in model_id.lower() and "glm-4.7" not in model_id.lower():
        data = base_data.copy()
        data["reasoning"] = {"effort": reasoning_effort}
        
        try:
            async with session.post(f"{API_URL}/chat/completions", headers=headers, json=data) as response:
                if response.status == 200:
                    try:
                        return (await response.json())['choices'][0]['message']['content']
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        print(f"Error parsing JSON from {model_id}: {e}")
                
                # If 400, assume it might be because the model doesn't support the reasoning param
                if response.status == 400:
                    print(f"Model {model_id} returned 400 with reasoning param. Retrying without...")
                else:
                    text = await response.text()
                    print(f"Error calling {model_id}: Status {response.status} - {text}")
                    # If it's a transient error (like 429 or 5xx), we'll fall through to retry without reasoning
                    if response.status not in [429, 500, 502, 503, 504]:
                        return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Exception calling {model_id} with reasoning: {e}")
            # Fall through to retry
            pass

    # Attempt 2: Retry without reasoning parameter
    try:
        async with session.post(f"{API_URL}/chat/completions", headers=headers, json=base_data) as response:
            if response.status == 200:
                try:
                    return (await response.json())['choices'][0]['message']['content']
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"Error parsing JSON from {model_id}: {e}")
                    return None
            
            text = await response.text()
            print(f"Error calling {model_id}: Status {response.status} - {text}")
            return None
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"Exception calling {model_id}: {e}")
        return None

async def process_model(session, model_name, model_id, prompt, default_extension, output_directory, semaphore):
    """
    Wrapper to process a single model: call, extract, save.
    """
    async with semaphore:
        print(f"Calling {model_id} (for {model_name})...")
        try:
            content = await call_model_async(session, model_id, prompt)
            
            if content:
                parsed = await parse_with_gemini(session, content)
                
                if parsed and parsed.get('code'):
                    code = parsed['code']
                    # specific extension from parser or fallback
                    ext = parsed.get('extension', default_extension)
                    
                    final_filename = get_unique_filename(output_directory, model_name, ext)
                    out_filename = os.path.join(output_directory, final_filename)
                    
                    with open(out_filename, 'w', encoding='utf-8') as f:
                        f.write(code)
                    print(f"Saved to {out_filename}")
                else:
                    print(f"Failed to parse content for {model_id}")
            else:
                print(f"Failed to generate for {model_id}")
        except asyncio.CancelledError:
            print(f"Task for {model_id} was cancelled.")
            raise
        except Exception as e:
            print(f"Error processing {model_id}: {e}")

async def main_async(args):
    # Determine basic settings
    user_prompt = args.prompt
    
    # Combine with system prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\nTask: {user_prompt}"
    
    # Default hint for naming if parser fails
    extension = '.py' if (args.prompt and ('.py' in args.prompt.lower() or 'python' in args.prompt.lower())) else '.html'
    
    targets = get_target_models()

    if not targets:
        print("No models found in mapping.")
        return

    print(f"\nStarting async generation for {len(targets)} models...\n")
    print(f"Prompt: {user_prompt}\n")
    
    # Limit concurrency to avoid hitting rate limits too hard and to be more orderly
    semaphore = asyncio.Semaphore(12)
    
    # Increase timeout for large model generations
    timeout = aiohttp.ClientTimeout(total=600) # 10 minutes total per request
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # Generate directory name based on the actual task
            dir_name = await get_directory_name(session, user_prompt)
            full_output_dir = os.path.join(OUTPUT_DIR, dir_name)
            
            if not os.path.exists(full_output_dir):
                os.makedirs(full_output_dir)
                
            print(f"Output directory: {full_output_dir}")
            
            tasks = [process_model(session, name, m_id, full_prompt, extension, full_output_dir, semaphore) for name, m_id in targets]
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("\nOperation cancelled by user. Cleaning up...")
            # We don't need to do much here as aiohttp session will close
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='LLM Benchmark Generator')
    parser.add_argument('--prompt', type=str, help='Custom prompt for the models')
    args = parser.parse_args()
    
    if not args.prompt:
        print("\n--- LLM Benchmark Generator ---")
        try:
            args.prompt = input("Enter your prompt for the models: ").strip()
            if not args.prompt:
                print("No prompt provided. Exiting.")
                return
        except KeyboardInterrupt:
            print("\nExiting.")
            return

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        # This will be caught when asyncio.run(main_async(args)) is interrupted
        pass

if __name__ == "__main__":
    main()
