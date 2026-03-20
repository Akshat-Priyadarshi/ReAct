# ReAct: Synergizing Reasoning and Acting in Language Models

**Team:** Four of a kind
**Course:** Machine Learning (CS60050) | IIT Kharagpur

## Project Overview

This project presents an experimental evaluation of agent-based reasoning under compute constraints. We evaluate the vulnerabilities of standard Chain-of-Thought (CoT) prompting and baseline ReAct architectures on multi-hop and adversarial factual queries. To address these vulnerabilities, we introduce a **Memory-Augmented ReAct Agent** that utilizes a stateful conversation buffer to achieve meta-cognition, effectively preventing infinite API search loops and improving autonomous task resolution.

## Models and Infrastructure

### LLM Specifications

To thoroughly evaluate agent logic while managing token constraints, we utilized a two-tier model approach:

- **Primary Evaluation Model:** `llama-3.1-8b-instant` (8 Billion Parameters). Used to expose standard architectural flaws (e.g., error propagation, infinite looping) inherent in smaller parameter models without memory constraints.
- **High-Fidelity Model:** `llama-3.3-70b-versatile` (70 Billion Parameters).

### Compute and GPU Justification

We utilized **Kaggle Notebooks with GPU T4 x2 accelerators** for our environment setup, data pipeline generation, and potential scalability testing. However, to optimize our pipeline, primary experiments and model inference were executed via the **Groq API**. This offloaded the heavy computational load to highly optimized, external LPU (Language Processing Unit) clusters. This architectural choice allowed us to achieve zero-latency reasoning and focus entirely on advanced agent design without being bottlenecked by local institutional GPU compute limits.

## Methodology

We evaluated three distinct pipelines against a custom adversarial dataset designed to trigger hallucinations and search dead-ends:

1.  **Standard CoT (No Tools):** A baseline to measure the rate of hallucination when external grounding is unavailable.
2.  **Baseline ReAct:** An agent equipped with a Wikipedia API tool but lacking stateful memory.
3.  **Memory-Augmented ReAct:** An advanced agent utilizing `ConversationBufferMemory` to maintain a sliding window of historical thoughts and actions.

## Results and Analysis

| Model Architecture     | Exact Match (EM) Accuracy      | Pipeline Stability        |
| :--------------------- | :----------------------------- | :------------------------ |
| Standard CoT           | 1.00\* (Artificially Inflated) | Stable (No Tools)         |
| Baseline ReAct         | 0.50                           | Unstable (Infinite Loops) |
| Memory-Augmented ReAct | 0.67                           | Highly Stable             |

### Qualitative Insights (The Metric Loophole)

While the Standard CoT pipeline achieved a perfect quantitative score, qualitative log analysis revealed severe hallucinations and metric exploitation. For example, the CoT model hallucinated a marriage between Keanu Reeves and Alexandra Grant. Furthermore, it successfully guessed keywords like "Arthur's Magazine" in its failure outputs, inadvertently triggering false positives in the Exact Match (EM) metric.

### The Impact of the Memory Buffer

The Baseline ReAct agent struggled with "trap" queries (e.g., historical facts that do not exist). When Wikipedia returned ambiguous data, the agent lacked meta-cognition, resulting in verbose apologies and failure to output the definitive "unavailable" ground truth.

By injecting the memory buffer, the **Memory-Augmented ReAct** agent successfully tracked its historical API calls. When a search failed, the agent recognized the dead end, dynamically pivoted its search syntax, and ultimately concluded when information was genuinely unavailable, successfully breaking the infinite search loop and improving accuracy to 0.67. Both ReAct models encountered rigid string-matching penalties (e.g., outputting "directors" instead of the singular "director"), highlighting the limitations of standard NLP evaluation metrics for reasoning agents.
