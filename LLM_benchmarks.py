#!/usr/bin/env python3
"""
LLM Benchmark Generator

Benchmarks LLMs via the OpenRouter API, saves generated code
for comparison, and tracks lifetime performance analytics.
"""

import os
import sys
import json
import re
import shutil
import asyncio
import time
from datetime import datetime, timezone
import aiohttp
import argparse

try:
    import readline  # noqa: F401 — enables arrow-key/history editing in input()
except ImportError:
    pass


# ━━ Configuration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API_URL          = "https://openrouter.ai/api/v1"
OUTPUT_DIR       = "benchmarkResults"
HISTORY_FILE     = ".benchmark_history.json"
MAX_CONCURRENCY  = 12
REQUEST_TIMEOUT  = 600  # seconds

SYSTEM_PROMPT_CODE = (
    "You are an expert programmer. Your goal is to provide a complete, "
    "fully functional, single-file implementation based on the user's request. "
    "Do not include any external modules or dependencies. "
    "Return ONLY the code, with no preamble or explanation."
)

SYSTEM_PROMPT_TEXT = (
    "You are a knowledgeable assistant. Provide a clear, detailed, and "
    "well-structured answer to the user's question. Use Markdown formatting "
    "for readability. Do not include code unless the user explicitly asks for it."
)

MODEL_MAPPING = {
    #'claudeOpus4.5':   'anthropic/claude-opus-4.5',
    #'gemini3_0Flash':  'google/gemini-3-flash-preview',
    'gemini3_0Pro':     'google/gemini-3-pro-preview',
    #'gpt5_2':          'openai/gpt-5.2',
    #'deepseekV3_2':    'deepseek/deepseek-v3.2',
    #'mimoV2Flash':     'xiaomi/mimo-v2-flash',
    #'gpt5_2Codex':      'openai/gpt-5.2-codex',
    'kimik2_5':         'moonshotai/kimi-k2.5',
    'minimax_m2.5':     'minimax/minimax-m2.5',
    'glm5':             'z-ai/glm-5',
    'claudeOpus4.6':    'anthropic/claude-opus-4.6',
}

_PAD = max((len(n) for n in MODEL_MAPPING), default=12) + 4  # column width


# ━━ Terminal Styling ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR") is not None


class S:
    """ANSI escape codes — empty strings when color is disabled."""
    RST  = "" if _NO_COLOR else "\033[0m"
    BOLD = "" if _NO_COLOR else "\033[1m"
    DIM  = "" if _NO_COLOR else "\033[2m"
    RED  = "" if _NO_COLOR else "\033[31m"
    GRN  = "" if _NO_COLOR else "\033[32m"
    YEL  = "" if _NO_COLOR else "\033[33m"
    BLU  = "" if _NO_COLOR else "\033[34m"
    CYN  = "" if _NO_COLOR else "\033[36m"
    HRED = "" if _NO_COLOR else "\033[91m"
    HGRN = "" if _NO_COLOR else "\033[92m"
    HYEL = "" if _NO_COLOR else "\033[93m"
    HBLU = "" if _NO_COLOR else "\033[94m"
    HCYN = "" if _NO_COLOR else "\033[96m"
    HWHT = "" if _NO_COLOR else "\033[97m"


# ── Glyphs ────────────────────────────────────────────────────────────────────

_ok    = f"{S.HGRN}✓{S.RST}"
_fail  = f"{S.HRED}✗{S.RST}"
_wait  = f"{S.HYEL}●{S.RST}"
_work  = f"{S.HCYN}◌{S.RST}"
_skip  = f"{S.DIM}○{S.RST}"
_arrow = f"{S.DIM}→{S.RST}"
_dot   = f"{S.DIM}·{S.RST}"
_tri   = f"{S.DIM}▸{S.RST}"

_SPIN  = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# ━━ Progress Display ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProgressTracker:
    """Animated progress spinner that auto-coordinates with print() calls.

    While active every ``print()`` is intercepted to first clear the spinner
    line, so output never overlaps the animation.  The spinner resumes on
    the next tick (~80 ms).

    *results* should be the shared dict populated by ``process_model`` —
    ``len(results)`` is used as the live completion count.
    """

    def __init__(self, total, results, label="Generating"):
        self._total = total
        self._results = results
        self._label = label
        self._running = False
        self._task = None
        self._start = time.monotonic()
        self._is_tty = sys.stdout.isatty()
        self._original_print = None

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if not self._is_tty:
            return
        self._running = True
        self._start = time.monotonic()
        self._install_hook()
        self._task = asyncio.create_task(self._animate())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        self._uninstall_hook()

    # ── animation loop ───────────────────────────────────────────────────

    async def _animate(self):
        idx = 0
        try:
            while self._running:
                done = len(self._results)
                elapsed = format_duration(time.monotonic() - self._start)
                frame = _SPIN[idx % len(_SPIN)]
                text = (f"\r  {S.HCYN}{frame}{S.RST} "
                        f"{self._label}  "
                        f"{S.DIM}{done}/{self._total} complete · "
                        f"{elapsed}{S.RST}\033[K")
                sys.stdout.write(text)
                sys.stdout.flush()
                idx += 1
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            pass

    # ── print() hook ─────────────────────────────────────────────────────
    #    asyncio is single-threaded so no lock is needed: the spinner is
    #    always sleeping when another coroutine's print() fires.

    def _install_hook(self):
        import builtins
        self._original_print = builtins.print
        tracker = self
        original = self._original_print

        def _hooked(*args, **kwargs):
            if tracker._running:
                sys.stdout.write("\r\033[K")
            original(*args, **kwargs)

        builtins.print = _hooked

    def _uninstall_hook(self):
        import builtins
        if self._original_print is not None:
            builtins.print = self._original_print
            self._original_print = None


# ━━ Formatting Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _vlen(text):
    """Visible length of *text*, ignoring ANSI escape sequences."""
    return len(re.sub(r'\033\[[0-9;]*m', '', str(text)))


def _rpad(text, width):
    """Left-align *text* to *width*, ANSI-aware."""
    return text + " " * max(0, width - _vlen(text))


def _tw():
    """Terminal width, clamped to [60, 120]."""
    return max(60, min(120, shutil.get_terminal_size((80, 24)).columns))


def _rule(label="", heavy=False):
    """Print a horizontal rule with an optional section label."""
    w = _tw() - 4
    ch = "━" if heavy else "─"
    if label:
        vis = _vlen(label)
        seg = (ch * 2
               + f" {S.BOLD}{S.CYN}{label}{S.RST}{S.DIM} "
               + ch * max(1, w - vis - 4))
    else:
        seg = ch * w
    print(f"  {S.DIM}{seg}{S.RST}")


def format_duration(seconds):
    """Format *seconds* into a concise human-readable string."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.0f}s"


def _truncate(text, length=60):
    """Truncate *text* with an ellipsis if needed."""
    return text if len(text) <= length else text[:length - 1] + "…"


# ━━ Analytics Persistence ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _history_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), HISTORY_FILE)


def load_history():
    """Load the analytics history from disk."""
    path = _history_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
                if isinstance(data, dict) and "runs" in data:
                    return data
        except (json.JSONDecodeError, IOError):
            pass
    return {"version": 1, "runs": []}


def save_history(history):
    """Persist the analytics history to disk."""
    try:
        with open(_history_path(), "w") as fh:
            json.dump(history, fh, indent=2)
    except IOError as exc:
        print(f"    {_tri} {S.DIM}could not save history: {exc}{S.RST}")


def record_run(history, prompt, output_dir, total_time, model_results):
    """Append the results of a benchmark run to *history* and save."""
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "output_dir": output_dir or "",
        "total_time_s": round(total_time, 2),
        "models": {
            name: {
                "status": info["status"],
                "time_s": round(info["time_s"], 2),
                "file": info.get("file"),
            }
            for name, info in model_results.items()
        },
    }
    history["runs"].append(run)
    save_history(history)


# ━━ Analytics Display ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def display_analytics(history, compact=False):
    """Print lifetime model performance analytics."""
    runs = history.get("runs", [])
    if not runs:
        print(f"  {S.DIM}No history yet. Complete a run to begin tracking.{S.RST}")
        return

    # ── Aggregate per-model stats ──────────────────────────────────────────
    stats = {}
    for run in runs:
        for name, res in run.get("models", {}).items():
            if name not in stats:
                stats[name] = {"runs": 0, "ok": 0, "fail": 0,
                               "cancel": 0, "times": []}
            s = stats[name]
            s["runs"] += 1
            status = res.get("status", "failed")
            if status == "success":
                s["ok"] += 1
                t = res.get("time_s")
                if t is not None:
                    s["times"].append(t)
            elif status == "cancelled":
                s["cancel"] += 1
            else:
                s["fail"] += 1

    # Sort: success-rate desc, then average time asc
    ranked = sorted(stats.items(), key=lambda x: (
        -(x[1]["ok"] / x[1]["runs"] if x[1]["runs"] else 0),
        (sum(x[1]["times"]) / len(x[1]["times"])
         if x[1]["times"] else float("inf")),
    ))

    n = len(runs)
    col = max((len(name) for name, _ in ranked), default=12) + 2
    col = max(col, _PAD)

    # ── Section header ─────────────────────────────────────────────────────
    print()
    _rule(f"Lifetime Analytics ({n} run{'s' if n != 1 else ''})")
    print()

    # ── Table header ───────────────────────────────────────────────────────
    hdr = (f"  {S.BOLD}{'MODEL':<{col}}{'RUNS':>5}  {'RATE':>5}"
           f"  {'AVG':>8}  {'BEST':>8}  {'WORST':>8}{S.RST}")
    print(hdr)
    print(f"  {S.DIM}{'─' * min(col + 40, _tw() - 4)}{S.RST}")

    # ── Table rows ─────────────────────────────────────────────────────────
    total_calls = total_ok = 0
    all_times = []

    for name, s in ranked:
        rate = (s["ok"] / s["runs"] * 100) if s["runs"] else 0
        avg_v  = sum(s["times"]) / len(s["times"]) if s["times"] else None
        best_v = min(s["times"]) if s["times"] else None
        wrst_v = max(s["times"]) if s["times"] else None

        rate_s = f"{rate:>4.0f}%"
        if rate >= 90:
            rate_c = f"{S.HGRN}{rate_s}{S.RST}"
        elif rate >= 60:
            rate_c = f"{S.HYEL}{rate_s}{S.RST}"
        else:
            rate_c = f"{S.HRED}{rate_s}{S.RST}"

        print(f"  {name:<{col}}{s['runs']:>5}  {rate_c}"
              f"  {S.CYN}{format_duration(avg_v):>8}{S.RST}"
              f"  {S.GRN}{format_duration(best_v):>8}{S.RST}"
              f"  {S.DIM}{format_duration(wrst_v):>8}{S.RST}")

        total_calls += s["runs"]
        total_ok += s["ok"]
        all_times.extend(s["times"])

    # ── Totals ─────────────────────────────────────────────────────────────
    overall = (total_ok / total_calls * 100) if total_calls else 0
    avg_all = format_duration(
        sum(all_times) / len(all_times) if all_times else None
    )
    print()
    print(f"  {total_calls} calls {_dot} {total_ok} passed {_dot} "
          f"{S.BOLD}{overall:.0f}%{S.RST} {_dot} avg {S.CYN}{avg_all}{S.RST}")

    # ── Recent prompts (full view only) ────────────────────────────────────
    if not compact and runs:
        print()
        _rule("Recent Prompts")
        print()
        for run in reversed(runs[-8:]):
            ts = run.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts)
                date_s = dt.strftime("%b %d %H:%M")
            except (ValueError, TypeError):
                date_s = "—"
            models = run.get("models", {})
            ok = sum(1 for r in models.values()
                     if r.get("status") == "success")
            tot = len(models)
            prompt = _truncate(run.get("prompt", "—"), _tw() - 26)
            print(f"  {S.DIM}{date_s}{S.RST}  {ok}/{tot}  {prompt}")


# ━━ File Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_unique_filename(directory, base_name, extension):
    """Return a unique filename, appending _v2, _v3, … if needed."""
    base = re.sub(r'cursor', '', base_name, flags=re.IGNORECASE)
    if not extension.startswith('.'):
        extension = f".{extension}"

    path = os.path.join(directory, f"{base}{extension}")
    if not os.path.exists(path):
        return f"{base}{extension}"

    counter = 2
    while True:
        name = f"{base}_v{counter}{extension}"
        if not os.path.exists(os.path.join(directory, name)):
            return name
        counter += 1


# ━━ API Key ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_api_key():
    """Load API key from environment or .env file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    if os.path.exists(".env"):
        try:
            with open(".env", "r") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "OPENROUTER_API_KEY":
                            return v.strip().strip('"').strip("'")
        except Exception as exc:
            print(f"    {_tri} {S.DIM}could not read .env: {exc}{S.RST}")

    return None


# ━━ API Calls ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def call_model_async(session, api_key, model_id, prompt,
                           reasoning_effort="high"):
    """Call the OpenRouter API for a specific model."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Benchmark Script",
    }

    base_data = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200000,
    }

    # Gemini 3 models require temperature = 1
    if "gemini-3" in model_id.lower():
        base_data["temperature"] = 1.0

    # Attempt 1 — with reasoning (skip for unsupported models)
    skip_reasoning = ("mimo" in model_id.lower()
                      or "glm-4.7" in model_id.lower())

    if reasoning_effort and not skip_reasoning:
        data = {**base_data, "reasoning": {"effort": reasoning_effort}}
        try:
            async with session.post(
                f"{API_URL}/chat/completions",
                headers=headers, json=data,
            ) as resp:
                if resp.status == 200:
                    try:
                        body = await resp.json()
                        return body['choices'][0]['message']['content']
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        print(f"    {_tri} {S.DIM}parse error "
                              f"({model_id}): {e}{S.RST}")

                if resp.status == 400:
                    print(f"    {_tri} {S.DIM}{model_id} 400 w/ reasoning"
                          f" — retrying without…{S.RST}")
                else:
                    text = await resp.text()
                    print(f"    {_tri} {S.DIM}{model_id}: {resp.status}"
                          f" — {text[:120]}{S.RST}")
                    if resp.status not in (429, 500, 502, 503, 504):
                        return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"    {_tri} {S.DIM}{model_id} reasoning err: "
                  f"{exc}{S.RST}")

    # Attempt 2 — without reasoning
    try:
        async with session.post(
            f"{API_URL}/chat/completions",
            headers=headers, json=base_data,
        ) as resp:
            if resp.status == 200:
                try:
                    body = await resp.json()
                    return body['choices'][0]['message']['content']
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"    {_tri} {S.DIM}parse error "
                          f"({model_id}): {e}{S.RST}")
                    return None

            text = await resp.text()
            print(f"    {_tri} {S.DIM}{model_id}: {resp.status}"
                  f" — {text[:120]}{S.RST}")
            return None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"    {_tri} {S.DIM}{model_id}: {exc}{S.RST}")
        return None


async def get_directory_name(session, api_key, prompt):
    """Use a fast model to derive a short directory name from the prompt."""
    model = "google/gemini-2.5-flash"
    naming_prompt = (
        "Generate a short, concise, snake_case directory name (max 3 words) "
        "that summarizes the following prompt. Return ONLY the directory "
        "name, no other text, no markdown formatting.\n\nPrompt: " + prompt
    )

    try:
        name = await call_model_async(session, api_key, model, naming_prompt)
        if name:
            clean = name.strip().replace('`', '').strip()
            clean = "".join(c for c in clean if c.isalnum() or c in "._- ")
            if clean:
                return clean
    except Exception as exc:
        print(f"    {_tri} {S.DIM}dir name error: {exc}{S.RST}")

    return "benchmark_output"


async def parse_with_gemini(session, api_key, model_name, content):
    """Use a fast model to extract code, language, and extension."""
    model = "google/gemini-2.5-flash"
    parsing_prompt = (
        "You are a code extraction engine. Your task is to extract the "
        "executable code from the provided text.\n"
        "1. Identify the programming language.\n"
        "2. Extract the FULL code content without any markdown formatting.\n"
        "3. Determine the correct file extension (e.g., .html, .py, .js).\n"
        "Return a raw JSON object with keys: 'code', 'extension', 'language'. "
        "Do not include any markdown formatting or explanation.\n\n"
        f"Text to parse:\n{content[:50000]}"
    )

    try:
        response = await call_model_async(
            session, api_key, model, parsing_prompt,
            reasoning_effort="minimal",
        )
        if not response:
            return None

        clean = response.strip()
        if "```" in clean:
            match = re.search(
                r'```(?:json)?\s*(.*?)\s*```', clean, re.DOTALL)
            if match:
                clean = match.group(1)

        return json.loads(clean)
    except Exception as exc:
        print(f"    {_tri} {S.DIM}parse error for "
              f"{model_name}: {exc}{S.RST}")
        return None


# ━━ Model Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def process_model(session, api_key, model_name, model_id, prompt,
                        default_ext, output_dir_task, semaphore, results):
    """Generate code from a single model, parse it, and save to disk."""
    start = time.monotonic()
    print(f"  {_wait} {model_name:<{_PAD}}  "
          f"{S.DIM}calling {model_id}…{S.RST}")

    try:
        async with semaphore:
            content = await call_model_async(
                session, api_key, model_id, prompt)

        elapsed = time.monotonic() - start

        if not content:
            print(f"  {_fail} {model_name:<{_PAD}}  "
                  f"{S.RED}no response{S.RST}  "
                  f"{S.DIM}[{format_duration(elapsed)}]{S.RST}")
            results[model_name] = {
                "status": "failed", "time_s": elapsed, "file": None}
            return

        print(f"  {_work} {model_name:<{_PAD}}  "
              f"{S.DIM}parsing…{S.RST}")
        parsed = await parse_with_gemini(
            session, api_key, model_name, content)

        if parsed and parsed.get("code"):
            ext = parsed.get("extension", default_ext)
            output_dir = await output_dir_task
            filename = get_unique_filename(output_dir, model_name, ext)
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(parsed["code"])

            elapsed = time.monotonic() - start
            print(f"  {_ok} {S.BOLD}{model_name:<{_PAD}}{S.RST}  "
                  f"saved {_arrow} {S.GRN}{filename}{S.RST}  "
                  f"{S.DIM}[{format_duration(elapsed)}]{S.RST}")
            results[model_name] = {
                "status": "success", "time_s": elapsed, "file": filename}
        else:
            elapsed = time.monotonic() - start
            print(f"  {_fail} {model_name:<{_PAD}}  "
                  f"{S.RED}parse failed{S.RST}  "
                  f"{S.DIM}[{format_duration(elapsed)}]{S.RST}")
            results[model_name] = {
                "status": "failed", "time_s": elapsed, "file": None}

    except asyncio.CancelledError:
        elapsed = time.monotonic() - start
        print(f"  {_skip} {model_name:<{_PAD}}  "
              f"{S.DIM}cancelled  [{format_duration(elapsed)}]{S.RST}")
        results[model_name] = {
            "status": "cancelled", "time_s": elapsed, "file": None}
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"  {_fail} {model_name:<{_PAD}}  "
              f"{S.RED}{exc}{S.RST}  "
              f"{S.DIM}[{format_duration(elapsed)}]{S.RST}")
        results[model_name] = {
            "status": "failed", "time_s": elapsed, "file": None}


async def process_model_text(session, api_key, model_name, model_id, prompt,
                             output_dir_task, semaphore, results):
    """Query a single model for a text response and save as Markdown."""
    start = time.monotonic()
    print(f"  {_wait} {model_name:<{_PAD}}  "
          f"{S.DIM}calling {model_id}…{S.RST}")

    try:
        async with semaphore:
            content = await call_model_async(
                session, api_key, model_id, prompt)

        elapsed = time.monotonic() - start

        if not content:
            print(f"  {_fail} {model_name:<{_PAD}}  "
                  f"{S.RED}no response{S.RST}  "
                  f"{S.DIM}[{format_duration(elapsed)}]{S.RST}")
            results[model_name] = {
                "status": "failed", "time_s": elapsed, "file": None}
            return

        # Save raw response directly as Markdown — no parsing step
        output_dir = await output_dir_task
        filename = get_unique_filename(output_dir, model_name, ".md")
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)

        elapsed = time.monotonic() - start
        print(f"  {_ok} {S.BOLD}{model_name:<{_PAD}}{S.RST}  "
              f"saved {_arrow} {S.GRN}{filename}{S.RST}  "
              f"{S.DIM}[{format_duration(elapsed)}]{S.RST}")
        results[model_name] = {
            "status": "success", "time_s": elapsed, "file": filename}

    except asyncio.CancelledError:
        elapsed = time.monotonic() - start
        print(f"  {_skip} {model_name:<{_PAD}}  "
              f"{S.DIM}cancelled  [{format_duration(elapsed)}]{S.RST}")
        results[model_name] = {
            "status": "cancelled", "time_s": elapsed, "file": None}
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"  {_fail} {model_name:<{_PAD}}  "
              f"{S.RED}{exc}{S.RST}  "
              f"{S.DIM}[{format_duration(elapsed)}]{S.RST}")
        results[model_name] = {
            "status": "failed", "time_s": elapsed, "file": None}


# ━━ Main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main_async(args, api_key):
    user_prompt = args.prompt
    text_mode = getattr(args, "text", False)

    if text_mode:
        sys_prompt = SYSTEM_PROMPT_TEXT
        full_prompt = f"{sys_prompt}\n\nQuestion: {user_prompt}"
        default_ext = ".md"
    else:
        sys_prompt = SYSTEM_PROMPT_CODE
        full_prompt = f"{sys_prompt}\n\nTask: {user_prompt}"
        default_ext = (
            ".py" if "python" in user_prompt.lower()
            or ".py" in user_prompt.lower()
            else ".html"
        )

    targets = list(MODEL_MAPPING.items())
    if not targets:
        print(f"  {_fail} No models configured in MODEL_MAPPING.")
        return

    mode_label = f"{S.HYEL}TEXT{S.RST}" if text_mode else f"{S.HCYN}CODE{S.RST}"

    # ── Config display ─────────────────────────────────────────────────────
    _rule(heavy=True)
    print()
    print(f"  {S.DIM}{'MODE':>8}{S.RST}  {mode_label}")
    print(f"  {S.DIM}{'PROMPT':>8}{S.RST}  "
          f"{S.BOLD}{_truncate(user_prompt, _tw() - 14)}{S.RST}")
    print(f"  {S.DIM}{'MODELS':>8}{S.RST}  {len(targets)} active")
    print()
    _rule()
    print()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=30)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    results = {}
    output_dir_final = [None]
    t0 = time.monotonic()

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector,
    ) as session:
        tracker = ProgressTracker(len(targets), results)
        try:
            async def resolve_output_dir():
                dir_name = await get_directory_name(
                    session, api_key, user_prompt)
                out = os.path.join(OUTPUT_DIR, dir_name)
                os.makedirs(out, exist_ok=True)

                pf = os.path.join(out, "prompt.txt")
                if not os.path.exists(pf):
                    with open(pf, "w", encoding="utf-8") as fh:
                        fh.write(user_prompt)

                output_dir_final[0] = out
                print(f"  {S.DIM}{'OUTPUT':>8}{S.RST}  {out}")
                return out

            output_dir_task = asyncio.create_task(resolve_output_dir())
            await tracker.start()

            if text_mode:
                tasks = [
                    process_model_text(
                        session, api_key, name, mid, full_prompt,
                        output_dir_task, semaphore, results,
                    )
                    for name, mid in targets
                ]
            else:
                tasks = [
                    process_model(
                        session, api_key, name, mid, full_prompt,
                        default_ext, output_dir_task, semaphore, results,
                    )
                    for name, mid in targets
                ]
            await asyncio.gather(*tasks, return_exceptions=True)

            if not output_dir_task.done():
                output_dir_task.cancel()

        except asyncio.CancelledError:
            print(f"\n  {S.DIM}Cancelled.{S.RST}")
        except Exception as exc:
            print(f"\n  {_fail} {S.RED}{exc}{S.RST}")
        finally:
            await tracker.stop()

    # ── Run results ────────────────────────────────────────────────────────
    total_time = time.monotonic() - t0
    ok   = sum(1 for v in results.values() if v["status"] == "success")
    fail = sum(1 for v in results.values() if v["status"] == "failed")
    canc = sum(1 for v in results.values() if v["status"] == "cancelled")

    print()
    _rule("Run Results")
    print()

    # Ranked leaderboard — successes first (fastest wins), then failures
    def _rank_key(item):
        _, v = item
        order = {"success": 0, "failed": 1, "cancelled": 2}
        return (order.get(v["status"], 3), v["time_s"])

    for i, (name, info) in enumerate(
        sorted(results.items(), key=_rank_key), 1
    ):
        st = info["status"]
        t = format_duration(info["time_s"])

        if st == "success":
            sym = _ok
            detail = f"saved {_arrow} {S.GRN}{info['file']}{S.RST}"
        elif st == "cancelled":
            sym = _skip
            detail = f"{S.DIM}cancelled{S.RST}"
        else:
            sym = _fail
            detail = f"{S.RED}failed{S.RST}"

        rank = f"{S.DIM}{i:>2}.{S.RST}"
        print(f"  {rank} {sym} {_rpad(name, _PAD)}"
              f"  {_rpad(detail, 34)}  {S.DIM}{t}{S.RST}")

    print()
    parts = []
    if ok:   parts.append(f"{S.HGRN}{ok} passed{S.RST}")
    if fail: parts.append(f"{S.HRED}{fail} failed{S.RST}")
    if canc: parts.append(f"{S.DIM}{canc} cancelled{S.RST}")
    parts.append(f"{format_duration(total_time)} total")
    print(f"  {f' {_dot} '.join(parts)}")

    # ── Record run & show lifetime analytics ───────────────────────────────
    history = load_history()
    record_run(history, user_prompt, output_dir_final[0],
               total_time, results)
    display_analytics(history, compact=True)

    print()
    _rule(heavy=True)
    print()


# ━━ Entry Point ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs via OpenRouter and track analytics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, help="Prompt to send to all models")
    parser.add_argument(
        "--text", action="store_true",
        help="Text mode: get prose answers instead of code")
    parser.add_argument(
        "--stats", action="store_true",
        help="Show lifetime analytics and exit")
    parser.add_argument(
        "--clear-history", action="store_true",
        help="Reset analytics history")
    args = parser.parse_args()

    # ── Stats-only mode ────────────────────────────────────────────────────
    if args.stats:
        print()
        _rule("LLM BENCHMARK", heavy=True)
        history = load_history()
        display_analytics(history, compact=False)
        print()
        _rule(heavy=True)
        print()
        return

    # ── Clear history ──────────────────────────────────────────────────────
    if args.clear_history:
        path = _history_path()
        if os.path.exists(path):
            os.remove(path)
            print(f"\n  {_ok} History cleared.\n")
        else:
            print(f"\n  {S.DIM}No history to clear.{S.RST}\n")
        return

    # ── API key ────────────────────────────────────────────────────────────
    api_key = load_api_key()
    if not api_key:
        print(f"\n  {_fail} {S.BOLD}OPENROUTER_API_KEY{S.RST} not set.")
        print(f"     Set via environment variable or .env file.\n")
        sys.exit(1)

    # ── Interactive prompt ─────────────────────────────────────────────────
    if not args.prompt:
        print()
        _rule("LLM BENCHMARK", heavy=True)
        print()

        # ── Mode selection (skip if --text was passed on CLI) ─────────
        if not args.text:
            print(f"  {S.DIM}Select mode:{S.RST}  "
                  f"{S.HCYN}[1]{S.RST} Code  "
                  f"{S.HYEL}[2]{S.RST} Text")
            try:
                mode_input = input(
                    f"  \001{S.DIM}\002mode\001{S.RST}\002 "
                    f"\001{S.HCYN}\002›\001{S.RST}\002 ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n  {S.DIM}Interrupted.{S.RST}\n")
                return
            if mode_input == "2":
                args.text = True
            print()

        try:
            # \001 / \002 tell readline that enclosed chars are non-printing
            # so cursor-position math stays correct with ANSI colours.
            _rl_prompt = (f"  \001{S.HCYN}\002›\001{S.RST}\002 ")
            args.prompt = input(_rl_prompt).strip()
            if not args.prompt:
                print(f"  {S.DIM}No prompt provided.{S.RST}\n")
                return
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {S.DIM}Interrupted.{S.RST}\n")
            return
        print()
    else:
        print()

    try:
        asyncio.run(main_async(args, api_key))
    except KeyboardInterrupt:
        print(f"\n\n  {S.DIM}Interrupted.{S.RST}\n")


if __name__ == "__main__":
    main()
