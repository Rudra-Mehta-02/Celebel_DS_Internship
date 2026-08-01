"""
Shared test fixtures and configuration.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_html():
    """Sample HTML for testing scraper and cleaner."""
    return """
    <html>
    <head><title>Test Article</title>
    <meta name="description" content="A test article about AI">
    </head>
    <body>
    <nav>Navigation menu here</nav>
    <article>
        <h1>Introduction to Machine Learning</h1>
        <p>Machine learning is a subset of artificial intelligence that focuses
        on building systems that learn from data. It has become one of the most
        important fields in computer science.</p>
        <p>There are three main types of machine learning:</p>
        <ul>
            <li>Supervised learning</li>
            <li>Unsupervised learning</li>
            <li>Reinforcement learning</li>
        </ul>
        <p>Deep learning, a subfield of machine learning, uses neural networks
        with many layers to model complex patterns in data.</p>
    </article>
    <footer>Copyright 2024</footer>
    </body>
    </html>
    """


@pytest.fixture
def sample_text():
    """Sample cleaned text for testing chunker."""
    return """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building
systems that learn from data. It has become one of the most important fields in
computer science. The field encompasses various algorithms and techniques that
enable computers to improve their performance on tasks through experience.

There are three main types of machine learning. Supervised learning involves
training models on labeled data where the correct output is known. Unsupervised
learning discovers hidden patterns in unlabeled data without explicit guidance.
Reinforcement learning trains agents to make decisions by rewarding desired
behaviors and penalizing undesired ones.

Deep learning is a subfield of machine learning that uses neural networks with
many layers to model complex patterns in data. Convolutional neural networks
are particularly effective for image recognition tasks. Recurrent neural networks
excel at processing sequential data like text and time series.

Transfer learning allows models trained on one task to be fine-tuned for related
tasks, significantly reducing the amount of training data needed. This technique
has been particularly successful in natural language processing with models like
BERT and GPT.

The transformer architecture has revolutionized natural language processing by
introducing the self-attention mechanism. This allows the model to weigh the
importance of different parts of the input when producing each part of the output.
Large language models like GPT-4 and Llama are based on this architecture."""


@pytest.fixture
def sample_chunks():
    """Pre-chunked text for testing vector store and RAG pipeline."""
    return [
        {
            "text": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "metadata": {"source": "test.txt", "title": "ML Intro", "chunk_index": 0},
        },
        {
            "text": "Deep learning uses neural networks with many layers to model complex patterns in data.",
            "metadata": {"source": "test.txt", "title": "ML Intro", "chunk_index": 1},
        },
        {
            "text": "The transformer architecture introduced the self-attention mechanism for natural language processing.",
            "metadata": {"source": "test.txt", "title": "ML Intro", "chunk_index": 2},
        },
    ]


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_llm():
    """Mock LLM that returns predictable responses."""
    mock = MagicMock()
    mock.generate.return_value = "This is a test response from the LLM."
    mock.generate_json.return_value = {
        "entities": [
            {"name": "Machine Learning", "type": "CONCEPT", "description": "A subset of AI"},
            {"name": "Neural Network", "type": "TECHNOLOGY", "description": "Computing system"},
        ],
        "relations": [
            {"source": "Machine Learning", "target": "Neural Network", "type": "USES", "description": "ML uses NNs"},
        ],
    }
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def mock_settings():
    """Mock settings for testing without .env file."""
    with patch.dict(os.environ, {
        "GROQ_API_KEY": "test_key_groq",
        "GOOGLE_API_KEY": "test_key_gemini",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "50",
        "RAG_TOP_K": "5",
        "HYBRID_ALPHA": "0.7",
        "MAX_HISTORY_TURNS": "10",
        "MAX_MEMORY_FACTS": "50",
    }):
        yield
