"""
Dynamic tools — 12 free/keyless tools for real-time data.

ALL tools are free and require NO API keys. They use public APIs,
local computation, or keyless services.
"""

from __future__ import annotations

import ast
import datetime
import json
import logging
import operator
import subprocess
import sys
import textwrap
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tool Registry ────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict] = {}


def register_tool(name: str, description: str):
    """Decorator to register a tool function."""
    def decorator(func):
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "function": func,
        }
        return func
    return decorator


def execute_tool(name: str, args: str = "") -> str:
    """Execute a registered tool by name."""
    if name not in TOOL_REGISTRY:
        return f"Unknown tool: {name}. Available: {', '.join(TOOL_REGISTRY.keys())}"
    try:
        result = TOOL_REGISTRY[name]["function"](args)
        return str(result)
    except Exception as e:
        logger.error("Tool '%s' failed: %s", name, e)
        return f"Tool error: {e}"


def get_tool_descriptions() -> str:
    """Return formatted descriptions of all tools for the router prompt."""
    lines = []
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"  - {name}: {info['description']}")
    return "\n".join(lines)


# ── Tool Implementations ────────────────────────────────────

@register_tool("get_datetime", "Get current date, time, and timezone")
def tool_datetime(args: str = "") -> str:
    now = datetime.datetime.now()
    utc = datetime.datetime.utcnow()
    return json.dumps({
        "local_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_time": utc.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timezone": datetime.datetime.now().astimezone().tzname(),
    })


@register_tool("web_search", "Search the web using DuckDuckGo (no API key)")
def tool_web_search(query: str = "") -> str:
    if not query:
        return "Please provide a search query."
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"• {r.get('title', 'N/A')}\n  {r.get('body', '')}\n  URL: {r.get('href', '')}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search failed: {e}"


@register_tool("wikipedia", "Look up a topic on Wikipedia")
def tool_wikipedia(query: str = "") -> str:
    if not query:
        return "Please provide a topic to look up."
    try:
        import wikipedia
        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(query, auto_suggest=True)
            summary = wikipedia.summary(query, sentences=5)
            return f"**{page.title}**\n\n{summary}\n\nURL: {page.url}"
        except wikipedia.DisambiguationError as e:
            options = e.options[:5]
            return f"Disambiguation: did you mean one of these?\n" + "\n".join(f"  - {o}" for o in options)
        except wikipedia.PageError:
            return f"No Wikipedia page found for '{query}'."
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


@register_tool("get_weather", "Get current weather for a city (wttr.in, no API key)")
def tool_weather(city: str = "") -> str:
    if not city:
        return "Please provide a city name."
    try:
        import requests
        resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current_condition", [{}])[0]
        return json.dumps({
            "city": city,
            "temperature_c": current.get("temp_C", "N/A"),
            "temperature_f": current.get("temp_F", "N/A"),
            "condition": current.get("weatherDesc", [{}])[0].get("value", "N/A"),
            "humidity": current.get("humidity", "N/A") + "%",
            "wind_kmph": current.get("windspeedKmph", "N/A"),
            "feels_like_c": current.get("FeelsLikeC", "N/A"),
        })
    except Exception as e:
        return f"Weather lookup failed: {e}"


@register_tool("get_stock", "Get stock price (Yahoo Finance, no API key)")
def tool_stock(symbol: str = "") -> str:
    if not symbol:
        return "Please provide a stock ticker symbol (e.g., AAPL, GOOGL)."
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        return json.dumps({
            "symbol": symbol.upper(),
            "name": info.get("shortName", "N/A"),
            "price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "currency": info.get("currency", "USD"),
            "change_percent": info.get("regularMarketChangePercent", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        })
    except Exception as e:
        return f"Stock lookup failed: {e}"


@register_tool("get_crypto", "Get cryptocurrency price (CoinGecko, no API key)")
def tool_crypto(coin: str = "") -> str:
    if not coin:
        return "Please provide a cryptocurrency name (e.g., bitcoin, ethereum)."
    try:
        import requests
        coin_id = coin.lower().strip()
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true&include_market_cap=true",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if coin_id not in data:
            return f"Cryptocurrency '{coin}' not found. Use full name (e.g., 'bitcoin')."
        info = data[coin_id]
        return json.dumps({
            "coin": coin_id,
            "price_usd": info.get("usd", "N/A"),
            "change_24h": f"{info.get('usd_24h_change', 0):.2f}%",
            "market_cap": info.get("usd_market_cap", "N/A"),
        })
    except Exception as e:
        return f"Crypto lookup failed: {e}"


@register_tool("calculator", "Evaluate a math expression safely (no eval)")
def tool_calculator(expression: str = "") -> str:
    if not expression:
        return "Please provide a math expression (e.g., '2 + 3 * 4')."
    try:
        # Safe math evaluation using AST
        result = _safe_math_eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


# Safe math evaluator using AST (no eval/exec)
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_math_eval(expr: str) -> float:
    """Safely evaluate a math expression using AST parsing."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and right > 100:
                raise ValueError("Exponent too large")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(_eval(node.operand))
        else:
            raise ValueError(f"Unsupported expression: {type(node).__name__}")

    return _eval(tree)


@register_tool("get_news", "Get latest news headlines via RSS feeds")
def tool_news(topic: str = "") -> str:
    if not topic:
        return "Please provide a news topic."
    try:
        import requests
        from bs4 import BeautifulSoup
        # Use Google News RSS
        url = f"https://news.google.com/rss/search?q={topic}&hl=en"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml-xml")
        items = soup.find_all("item", limit=5)
        if not items:
            return f"No news found for '{topic}'."
        headlines = []
        for item in items:
            title = item.find("title")
            pub_date = item.find("pubDate")
            headlines.append(
                f"• {title.text if title else 'N/A'}"
                + (f"\n  Published: {pub_date.text}" if pub_date else "")
            )
        return f"Latest news for '{topic}':\n\n" + "\n\n".join(headlines)
    except Exception as e:
        return f"News lookup failed: {e}"


@register_tool("read_url", "Fetch and extract text from a URL")
def tool_read_url(url: str = "") -> str:
    if not url:
        return "Please provide a URL."
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0"
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Truncate to ~2000 chars
        if len(text) > 2000:
            text = text[:2000] + "\n\n[...truncated]"
        return text
    except Exception as e:
        return f"URL read failed: {e}"


@register_tool("python_exec", "Execute Python code in a sandboxed subprocess")
def tool_python_exec(code: str = "") -> str:
    if not code:
        return "Please provide Python code to execute."
    # Block dangerous imports
    blocked = ["os", "sys", "subprocess", "shutil", "pathlib", "socket", "requests", "http"]
    for b in blocked:
        if f"import {b}" in code or f"from {b}" in code:
            return f"Blocked: importing '{b}' is not allowed in sandbox."
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Execution timed out (5 second limit)."
    except Exception as e:
        return f"Execution error: {e}"


@register_tool("unit_convert", "Convert between units (length, weight, temperature, etc.)")
def tool_unit_convert(expression: str = "") -> str:
    if not expression:
        return "Please provide a conversion (e.g., '100 celsius to fahrenheit', '5 km to miles')."
    try:
        import pint
        ureg = pint.UnitRegistry()
        # Try to parse "X unit1 to unit2"
        parts = expression.lower().replace(" to ", " ").replace(" in ", " ").split()
        if len(parts) < 3:
            return "Format: '<value> <from_unit> to <to_unit>' (e.g., '100 km to miles')"
        value = float(parts[0])
        from_unit = parts[1]
        to_unit = parts[-1]
        quantity = ureg.Quantity(value, from_unit)
        result = quantity.to(to_unit)
        return f"{value} {from_unit} = {result.magnitude:.4f} {to_unit}"
    except Exception as e:
        return f"Conversion failed: {e}"


@register_tool("get_definition", "Look up word definitions (Free Dictionary API)")
def tool_definition(word: str = "") -> str:
    if not word:
        return "Please provide a word to define."
    try:
        import requests
        resp = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=10)
        if resp.status_code == 404:
            return f"No definition found for '{word}'."
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return f"No definition found for '{word}'."
        entry = data[0]
        result = f"**{entry.get('word', word)}**"
        if entry.get("phonetic"):
            result += f" ({entry['phonetic']})"
        result += "\n"
        for meaning in entry.get("meanings", [])[:3]:
            pos = meaning.get("partOfSpeech", "")
            result += f"\n*{pos}*:"
            for defn in meaning.get("definitions", [])[:2]:
                result += f"\n  - {defn.get('definition', '')}"
                if defn.get("example"):
                    result += f'\n    Example: "{defn["example"]}"'
        return result
    except Exception as e:
        return f"Definition lookup failed: {e}"
