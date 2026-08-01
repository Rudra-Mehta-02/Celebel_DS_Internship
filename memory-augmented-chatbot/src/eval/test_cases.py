"""
Test cases for the evaluation framework.

30+ curated test cases across:
  - Factual RAG questions
  - Relationship/KG questions
  - Dynamic/tool questions
  - Memory/personalisation questions
  - Multi-hop reasoning questions
"""

from __future__ import annotations

# Each test case: {question, ground_truth, category, expected_route}
TEST_CASES = [
    # ── Factual RAG Questions ────────────────────────────────
    {
        "question": "What is deep learning?",
        "ground_truth": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to learn representations of data.",
        "category": "rag",
        "expected_route": "rag",
    },
    {
        "question": "Explain the transformer architecture",
        "ground_truth": "The transformer architecture uses self-attention mechanisms to process sequences in parallel, introduced in the 'Attention is All You Need' paper.",
        "category": "rag",
        "expected_route": "rag",
    },
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "Retrieval-augmented generation (RAG) combines information retrieval with text generation, allowing language models to access external knowledge.",
        "category": "rag",
        "expected_route": "rag",
    },
    {
        "question": "How does backpropagation work?",
        "ground_truth": "Backpropagation computes gradients of the loss function with respect to each weight by applying the chain rule, propagating errors backward through the network.",
        "category": "rag",
        "expected_route": "rag",
    },
    {
        "question": "What are convolutional neural networks used for?",
        "ground_truth": "Convolutional neural networks (CNNs) are primarily used for image recognition and computer vision tasks, using convolutional layers to detect features.",
        "category": "rag",
        "expected_route": "rag",
    },
    {
        "question": "Explain the attention mechanism in machine learning",
        "ground_truth": "The attention mechanism allows models to focus on different parts of the input when generating each part of the output, computing weighted sums of value vectors.",
        "category": "rag",
        "expected_route": "rag",
    },
    {
        "question": "What is transfer learning?",
        "ground_truth": "Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a different task.",
        "category": "rag",
        "expected_route": "rag",
    },
    {
        "question": "How do generative adversarial networks work?",
        "ground_truth": "GANs consist of two neural networks — a generator and a discriminator — that compete against each other. The generator creates fake data while the discriminator tries to distinguish real from fake.",
        "category": "rag",
        "expected_route": "rag",
    },
    {
        "question": "What is reinforcement learning?",
        "ground_truth": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximise cumulative reward.",
        "category": "rag",
        "expected_route": "rag",
    },
    {
        "question": "Explain word embeddings",
        "ground_truth": "Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points.",
        "category": "rag",
        "expected_route": "rag",
    },

    # ── Relationship/KG Questions ────────────────────────────
    {
        "question": "What is the relationship between deep learning and neural networks?",
        "ground_truth": "Deep learning is a subset of machine learning that specifically uses deep neural networks with multiple layers.",
        "category": "kg",
        "expected_route": "hybrid",
    },
    {
        "question": "How is BERT related to the Transformer architecture?",
        "ground_truth": "BERT (Bidirectional Encoder Representations from Transformers) is based on the Transformer architecture, specifically using only the encoder component.",
        "category": "kg",
        "expected_route": "hybrid",
    },
    {
        "question": "What concepts are connected to natural language processing?",
        "ground_truth": "NLP is connected to machine learning, deep learning, transformers, word embeddings, text classification, sentiment analysis, and language models.",
        "category": "kg",
        "expected_route": "kg",
    },
    {
        "question": "Which techniques are used in computer vision?",
        "ground_truth": "Computer vision uses CNNs, image classification, object detection, image segmentation, transfer learning, and data augmentation.",
        "category": "kg",
        "expected_route": "kg",
    },
    {
        "question": "What is the connection between GPT and large language models?",
        "ground_truth": "GPT (Generative Pre-trained Transformer) is one of the most prominent large language model families, based on the Transformer decoder architecture.",
        "category": "kg",
        "expected_route": "hybrid",
    },

    # ── Dynamic/Tool Questions ───────────────────────────────
    {
        "question": "What is the weather in London right now?",
        "ground_truth": "",
        "category": "tool",
        "expected_route": "tool",
    },
    {
        "question": "What is 145 * 37 + 892?",
        "ground_truth": "6257",
        "category": "tool",
        "expected_route": "tool",
    },
    {
        "question": "What is the current price of Bitcoin?",
        "ground_truth": "",
        "category": "tool",
        "expected_route": "tool",
    },
    {
        "question": "What date is it today?",
        "ground_truth": "",
        "category": "tool",
        "expected_route": "tool",
    },
    {
        "question": "Search the web for latest AI news",
        "ground_truth": "",
        "category": "tool",
        "expected_route": "tool",
    },

    # ── Memory/Personalisation Questions ─────────────────────
    {
        "question": "What programming language should I use for my next project?",
        "ground_truth": "Based on user preferences",
        "category": "memory",
        "expected_route": "direct",
        "planted_facts": [
            "User's favorite programming language is Python",
            "User works as a data scientist",
        ],
    },
    {
        "question": "Suggest a machine learning framework for me",
        "ground_truth": "Based on user preferences",
        "category": "memory",
        "expected_route": "rag",
        "planted_facts": [
            "User prefers Python",
            "User is interested in deep learning",
            "User works on NLP projects",
        ],
    },
    {
        "question": "What topics should I study next?",
        "ground_truth": "Based on user interests",
        "category": "memory",
        "expected_route": "direct",
        "planted_facts": [
            "User is learning about transformers",
            "User is interested in reinforcement learning",
            "User's goal is to become an ML engineer",
        ],
    },
    {
        "question": "Do you remember what I told you about my work?",
        "ground_truth": "Should recall user's work information",
        "category": "memory",
        "expected_route": "direct",
        "planted_facts": [
            "User works at Google",
            "User is a senior engineer",
        ],
    },
    {
        "question": "Based on my interests, explain autoencoders",
        "ground_truth": "Should personalise the explanation based on user background",
        "category": "memory",
        "expected_route": "rag",
        "planted_facts": [
            "User has a background in statistics",
            "User prefers mathematical explanations",
        ],
    },

    # ── Multi-hop Reasoning ──────────────────────────────────
    {
        "question": "Compare CNNs and RNNs — which is better for text classification?",
        "ground_truth": "While both can handle text, RNNs are traditionally used for sequential data. However, CNNs with 1D convolutions can also be effective for text classification and are often faster to train.",
        "category": "multi_hop",
        "expected_route": "hybrid",
    },
    {
        "question": "How has the attention mechanism influenced modern NLP?",
        "ground_truth": "The attention mechanism led to the Transformer architecture, which revolutionised NLP through models like BERT, GPT, and T5, enabling unprecedented performance on various tasks.",
        "category": "multi_hop",
        "expected_route": "hybrid",
    },
    {
        "question": "What are the pros and cons of using pre-trained language models?",
        "ground_truth": "Pros: reduced training time, better performance with less data, transfer learning capabilities. Cons: computational cost, potential bias, domain specificity issues.",
        "category": "multi_hop",
        "expected_route": "rag",
    },
    {
        "question": "Explain how gradient boosting differs from random forests",
        "ground_truth": "Random forests build trees independently in parallel (bagging), while gradient boosting builds trees sequentially, with each tree correcting the errors of the previous ones.",
        "category": "multi_hop",
        "expected_route": "rag",
    },
    {
        "question": "What role does feature engineering play in modern deep learning?",
        "ground_truth": "Deep learning has reduced the need for manual feature engineering as models can learn representations automatically, but feature engineering remains important for tabular data and domain-specific applications.",
        "category": "multi_hop",
        "expected_route": "hybrid",
    },
]


def get_test_cases(category: str | None = None) -> list[dict]:
    """Get test cases, optionally filtered by category."""
    if category:
        return [tc for tc in TEST_CASES if tc["category"] == category]
    return TEST_CASES


def get_memory_test_cases() -> list[dict]:
    """Get only test cases that include planted facts for memory testing."""
    return [tc for tc in TEST_CASES if tc.get("planted_facts")]
