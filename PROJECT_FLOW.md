# Project Journey: Teaching AI to Think, Act, and Remember

**Team:** Four of a kind (Akshat Priyadarshi, Hritwik Upadhyay, Ritabrata Sarkar, Harshit Singhal)  
**Institution:** IIT Kharagpur | **Course:** Machine Learning (CS60050)

---

## 1. The Starting Line: What Are We Trying to Do?

Imagine a brilliant scholar who has read millions of books but is now locked in a room without internet access. If you ask them a highly specific trivia question about an event from yesterday, they don't know the answer. However, because they are programmed to be helpful, they might just confidently make up a realistic-sounding lie.

In Artificial Intelligence, these scholars are called Large Language Models (LLMs). When they confidently make things up, it is called a **hallucination**.

Our goal was to give the AI a tool—specifically, access to search Wikipedia—and teach it how to use that tool to verify facts before answering our questions.

## 2. Phase 1: The Confident Liar (Chain-of-Thought)

First, we wanted to prove that the AI actually needed our help. We tested how it behaves _without_ tools using a method called "Chain-of-Thought." This simply means we told the AI: _"Think step-by-step before you answer."_

**The Difficulty:** The AI sounded incredibly smart, but its logic was flawed. We asked it a trick question: _"Who is the wife of the actor who played Neo in The Matrix?"_ The actor is Keanu Reeves, who is famously unmarried.

Instead of realizing he doesn't have a wife, the AI hallucinated that he was married to his partner, Alexandra Grant. If we weren't paying close attention, the AI's confident tone would have completely tricked us.

## 3. Phase 2: The Goldfish with a Smartphone (Baseline ReAct)

To fix the hallucination problem, we implemented a framework called **ReAct (Reasoning + Acting)** using a library called LangChain. We connected the AI to Wikipedia. Now, the AI's thought process looked like this:

_Think about the question → Search Wikipedia → Read the Observation → Think again._

**The Difficulties:**
This is where we hit massive engineering roadblocks.

1. **The API Crash:** Initially, our code kept crashing with "400 Errors." The AI was trying to type its Wikipedia search into the system using raw conversational text instead of the strict computer format (JSON) the system required. We had to upgrade our entire architecture to use modern "tool-calling" agents to force the AI to format its searches correctly.
2. **The Infinite Loop:** We fed the AI another trick question: _"Who was the childhood best friend of the oldest person alive in 1850?"_ The AI searched Wikipedia for the oldest person, then searched for their best friend. Wikipedia returned zero results. Because standard ReAct agents have no memory of what they _just_ did, the AI panicked and searched the exact same phrase again. And again. It entered an infinite loop, wasting time and computer processing power until the system forcefully shut it down.

The AI had tools, but it lacked **meta-cognition**—the ability to realize, _"I just tried that exact search, and it didn't work."_

## 4. Phase 3: The Breakthrough (Memory-Augmented Agent)

We realized we needed to give the AI a short-term memory buffer.

**The Solution:**
We injected a module called `ConversationBufferMemory` directly into the agent's code. We also wrote a strict rule in its core instructions: _Check your past actions. If a search fails, do not repeat it. Pivot to a new keyword. If you still can't find it, just tell us the information is unavailable._

**The Result:**
It worked perfectly. When faced with the same impossible 1850 historical question, the AI searched Wikipedia. When it got zero results, it looked at its own short-term memory, realized it hit a dead end, stopped searching, and correctly told us the information was "unavailable." The infinite loop was broken.

## 5. The Hardware Hurdle: Bypassing Compute Limits

To run AI models this advanced, you usually need massive, expensive graphics cards (GPUs).

**Our Solution:**
Instead of trying to run these massive models locally on our own laptops, we used **Kaggle Notebooks**, which provided us with a free cloud environment equipped with dual T4 GPUs. However, the real optimization was routing the AI's actual "thinking" through the **Groq API**. Groq processes AI models using highly specialized hardware called LPUs (Language Processing Units) in the cloud.

This allowed us to test our code rapidly using a smaller model (`Llama 3.1 8B`), and once our memory system successfully fixed the infinite loops, we seamlessly scaled up to a massive 70-billion parameter model (`Llama 3.3 70B`) without our local computers breaking a sweat.

## The Final Takeaway

Building an AI agent isn't just about plugging it into Google or Wikipedia. If an AI cannot remember its past failures, it will waste resources repeating them. By combining reasoning, acting, and a short-term memory buffer, we transformed a static, hallucinating LLM into a robust, cost-effective, and self-correcting autonomous agent.
