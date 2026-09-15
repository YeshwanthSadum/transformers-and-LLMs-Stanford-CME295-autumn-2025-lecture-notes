# Lecture 7: Agentic LLMs - Stanford CME 295

**Date:** November 14, 2025
**Instructors:** Afshine Amidi & Shervine Amidi (Adjunct Lecturers, Stanford)
**Course:** CME 295 - Transformers and Large Language Models, Autumn 2025

---

## 1. Recap and Motivation

Lecture 6 covered reasoning models and how allowing an LLM to produce a chain of thought before its final answer improves performance on tasks like math and coding. The key training algorithm was GRPO (Group Relative Policy Optimization), which computes advantages relative to a group of completions for the same prompt rather than training a separate value function. One practical finding was length bias: as RL training progresses, model outputs grow longer even after accuracy plateaus, because the loss formulation weights tokens differently depending on response length. Two mitigation strategies were discussed -- DAPO (which normalizes independently of sentence position) and "GRPO Done Right" (which removes the normalization term entirely).

With reasoning capabilities addressed, two gaps remain in vanilla LLMs. First, the model has no connection to knowledge that appeared after its pre-training cutoff. Second, the model cannot take actions in external systems. This lecture tackles both: Retrieval-Augmented Generation (RAG) connects the LLM to an evolving knowledge base, and tool calling plus agentic workflows let the LLM interact with the outside world.

---

## 2. Retrieval-Augmented Generation (RAG)

### 2.1 The Knowledge Cutoff Problem

Every LLM has a knowledge cutoff date -- the point after which the model has no information because its training data stopped there. GPT-5, for example, lists a knowledge cutoff of September 30, 2024. Any event after that date is invisible to the base model.

The obvious remedy -- continue training on newer data -- has serious drawbacks. Modifying an LLM's weights to inject new knowledge risks regression on previously learned capabilities. Moreover, if the base model has been fine-tuned for multiple downstream use cases, each fine-tuned variant would need to be re-updated, creating significant maintenance overhead. In practice, teams avoid additional training purely for knowledge injection.

### 2.2 Why Not Just Expand the Context Window?

A naive alternative is to dump all new information directly into the prompt. This runs into three problems:

**Context length is finite.** Models typically support on the order of hundreds of thousands of tokens. GPT-5's context window is 400,000 tokens, roughly equivalent to hundreds of pages. That is large but not enough to contain all new information indefinitely.

**The needle-in-a-haystack problem.** Even if context were unlimited, LLMs degrade when flooded with irrelevant information. The needle-in-a-haystack test places a single fact at varying positions within prompts of varying lengths and checks whether the model can retrieve it. Heat maps from GPT-4 experiments show that once the prompt exceeds a certain token count, retrieval accuracy drops, particularly when the fact is placed in the first half of the prompt. Flooding the context hurts performance even when the answer is technically present.

**Cost.** LLM API calls are priced per token. Larger input prompts cost more. At roughly one dollar per million tokens (GPT-5 order of magnitude), the expense accumulates quickly across many queries.

These constraints motivate a more selective approach: find only the relevant information and inject that into the prompt.

### 2.3 The RAG Framework

RAG stands for Retrieval-Augmented Generation. The name encodes its three stages:

1. **Retrieve** -- Given a user query, search a knowledge base to find documents (or document fragments) that are relevant to the query.
2. **Augment** -- Insert the retrieved information into the prompt alongside the original query. The prompt effectively becomes: "Here is background context: [retrieved text]. Now answer: [user question]."
3. **Generate** -- Feed the augmented prompt to the LLM to produce the response.

The critical observation is that the LLM receives the answer within its context -- the retrieval step is responsible for putting it there. If retrieval fails to surface the right documents, the entire pipeline fails. This is why most RAG engineering effort focuses on the retrieval stage.

---

## 3. Building the Knowledge Base

### 3.1 Chunking

The first step is to collect the documents that form the knowledge base and divide them into **chunks** -- fixed-length segments measured in tokens. Chunking is necessary because embeddings represent fixed-size inputs; a single embedding for an entire long document would be too coarse to capture local information.

Three hyperparameters govern chunking:

**Chunk size** determines how many tokens each chunk contains. Typical values are on the order of 500 tokens. If chunks are too small, individual chunks may lack sufficient context to be meaningful. If chunks are too large, their embeddings become diluted and may not represent the specific information within them well enough for similarity search.

**Overlap** controls how many tokens are shared between consecutive chunks. Typical values are in the low hundreds. Overlap exists because a naive split can sever a sentence or idea mid-thought. By repeating some tokens from the end of one chunk at the beginning of the next, each chunk retains enough surrounding context to be interpretable on its own.

**Embedding size** is the dimensionality of the vector that represents each chunk. Typical sizes are on the order of 1,500 dimensions. Larger embeddings can capture more nuance in complex documents but require more storage and computation at inference time.

The type of document also matters. Structured formats like JSON or Markdown have internal structure (headers, keys, nesting) that a purely token-count-based split can break. In practice, chunking strategies should be aware of document structure, though the lecture did not cover format-specific chunking in detail.

### 3.2 Computing Chunk Embeddings

Each chunk is passed through an encoder-only model (typically BERT-like) to produce a dense vector representation. A widely recommended starting point is the Sentence-BERT paper, which extends BERT to produce per-sequence embeddings optimized for similarity search. The training objective incentivizes high cosine similarity between embeddings of semantically related inputs and low cosine similarity between unrelated inputs.

You can either use a pre-trained embedding model off the shelf or train your own on domain-specific data. Pre-trained models are the common default.

---

## 4. Candidate Retrieval

Retrieval follows a two-stage pipeline borrowed from recommendation systems and information retrieval. The first stage, candidate retrieval, is a coarse filter that reduces millions of chunks down to roughly 100 potentially relevant candidates. The second stage, reranking, is a fine-grained scorer that produces the final top-K ordering. The two stages trade off speed against quality: the first must be fast (it touches every chunk in the knowledge base), while the second can afford to be slower (it only processes the shortlisted candidates).

### 4.1 Semantic Similarity Search (Bi-Encoder)

The standard approach represents both the query and each chunk as embeddings, then ranks chunks by cosine similarity to the query embedding:

$$\text{cosine similarity}(\mathbf{q}, \mathbf{c}) = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\| \, \|\mathbf{c}\|}$$

where $\mathbf{q}$ is the query embedding and $\mathbf{c}$ is the chunk embedding. Chunks with the highest similarity scores are retained as candidates.

This setup is called a **bi-encoder** because the query and the chunk each pass through an encoder independently. The two encoders may share weights or be separate. The independence is what makes the approach scalable: chunk embeddings are computed once during knowledge base construction and stored; at query time, only the query needs to be encoded, and similarity computation is a dot product.

For large knowledge bases, even linear scans over all chunk embeddings become expensive. **Approximate Nearest Neighbor (ANN)** methods -- such as those provided by libraries like FAISS or Annoy -- partition the embedding space during index construction so that retrieval avoids a brute-force comparison against every chunk.

The default similarity metric is cosine similarity, though some implementations use L2 distance. When all embeddings are normalized to unit length, these metrics are monotonically related and produce equivalent rankings.

### 4.2 Keyword-Based Search (BM25)

Semantic similarity search finds chunks with similar meaning regardless of whether any words overlap. But sometimes the user's query contains specific keywords that must appear in the retrieved documents. BM25 is a heuristic relevance score based on term-frequency overlap between the query and the document. It guarantees that retrieved chunks contain words from the query.

Consider a query asking where a specific teddy bear named "Cuddly" is located. Semantic search might return chunks about a different teddy bear named "Huggy" because the two names are semantically similar. BM25 would correctly prioritize chunks containing the exact word "Cuddly."

### 4.3 Hybrid Search

In practice, many systems combine semantic similarity and BM25 into a hybrid retrieval score. The semantic component captures meaning-based relevance, while BM25 ensures keyword fidelity. The optimal blend depends on the use case: highly technical or entity-rich domains (legal, medical, product catalogs) tend to benefit more from BM25's keyword matching.

### 4.4 HyDE (Hypothetical Document Embeddings)

A recurring issue with bi-encoder search is that queries and documents are fundamentally different kinds of text. A query is typically short and phrased as a question; a document is longer and declarative. If both are embedded by the same encoder, their representations may not be directly comparable in the shared embedding space.

HyDE addresses this asymmetry by inserting an extra LLM call before embedding. Given the user's query, the LLM generates a hypothetical (fake) document that would answer the query. This synthetic document is then embedded and used for similarity search instead of the raw query. Because the synthetic document is structurally similar to real documents in the knowledge base, the resulting embedding is more comparable.

HyDE does not always improve results and adds latency (one extra LLM call per query). It is best treated as an empirical option to try rather than a default.

An alternative solution is to train separate encoders for queries and documents, so each encoder specializes in its respective text type. This approach is less common in practice due to the maintenance burden of managing two models.

### 4.5 Contextual Chunking

Naive chunking can produce fragments that are incoherent when read in isolation. Contextual chunking addresses this by prepending a short context summary to each chunk. The process works as follows:

1. Take the full document and a specific chunk from that document.
2. Prompt an LLM: "Given this document and this chunk, provide a short context that helps make sense of this chunk."
3. Prepend the generated context to the chunk before computing its embedding.

This requires one LLM call per chunk, which can be expensive at scale. The cost is mitigated by **prompt caching**: because every call shares the same prefix (the full document), and decoder-only LLMs compute activations left-to-right, the activations for the shared prefix can be computed once and reused across all chunks of the same document.

Prompt caching is offered by major API providers. If you examine model pricing pages, you will find a separate (lower) price for cached input tokens -- typically around one-tenth of the regular input token price. The practical takeaway is to structure prompts so that the repeated content appears at the beginning, maximizing the cacheable prefix.

---

## 5. Reranking (Cross-Encoder)

The candidate retrieval stage optimizes for recall: cast a wide net to avoid missing relevant documents. The reranking stage optimizes for precision: given the shortlisted candidates, produce an accurate top-K ordering.

### 5.1 Cross-Encoder Architecture

Where the bi-encoder computes query and chunk embeddings independently, the **cross-encoder** feeds both the query and the chunk into a single encoder simultaneously. This allows the model to compute attention between query tokens and chunk tokens, capturing interactions that independent embeddings cannot represent. The cross-encoder outputs a single relevance score for the (query, chunk) pair.

The cross-encoder is more expensive per comparison because it must run a full forward pass for each (query, chunk) pair rather than a single dot product. This is why it is reserved for the second stage, where the candidate set has been reduced from millions to hundreds.

The Sentence-BERT documentation provides detailed guidance on training and using cross-encoders for reranking.

---

## 6. Retrieval Metrics

Evaluating retrieval quality requires metrics that account for both relevance and ranking position. The setup assumes ground-truth labels: for each query, we know which chunks are actually relevant. The retrieval system produces a ranked list of K chunks, and we measure how well this list matches the ground truth.

### 6.1 NDCG (Normalized Discounted Cumulative Gain)

NDCG measures ranking quality by rewarding systems that place relevant documents at higher positions:

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i + 1)}$$

where $\text{rel}_i$ is the relevance label of the document at rank $i$ (1 if relevant, 0 otherwise), and the denominator $\log_2(i + 1)$ is the discount factor that penalizes relevant documents placed at lower ranks. A relevant document at rank 1 contributes more to the score than the same document at rank 5.

To make the score comparable across queries with different numbers of relevant documents, DCG is normalized by the Ideal DCG (IDCG) -- the DCG score that a perfect ranking would achieve:

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

NDCG ranges from 0 to 1, where 1 means the system's ranking perfectly matches the optimal ranking. The normalization ensures that a query with 2 relevant documents and a query with 20 relevant documents are evaluated on the same scale.

### 6.2 MRR (Mean Reciprocal Rank)

MRR is a simpler metric that considers only the first relevant document in the ranked list:

$$\text{MRR} = \frac{1}{\text{rank of first relevant document}}$$

If the first relevant document appears at position 2, MRR is 0.5. If it appears at position 1, MRR is 1.0. MRR ignores all relevant documents after the first one, making it a coarser measure than NDCG. Despite this, it correlates well with user satisfaction in many search scenarios where finding one good result quickly is the primary goal.

### 6.3 Precision@K and Recall@K

These are direct adaptations of classification metrics to the ranking setting.

**Precision@K** asks: of the K documents I retrieved, how many are actually relevant?

$$\text{Precision@K} = \frac{|\{\text{relevant documents in top K}\}|}{K}$$

**Recall@K** asks: of all the documents that are actually relevant, how many did I retrieve in my top K?

$$\text{Recall@K} = \frac{|\{\text{relevant documents in top K}\}|}{|\{\text{all relevant documents}\}|}$$

### 6.4 Benchmarks

The **Massive Text Embedding Benchmark (MTEB)** is the standard benchmark for evaluating retrieval systems. It computes all of the above metrics across a variety of tasks and datasets, providing a comprehensive leaderboard for comparing embedding models and retrieval pipelines.

---

## 7. Tool Calling

### 7.1 From Unstructured to Structured Data

RAG addresses the knowledge gap by injecting unstructured text (documents, articles, web pages) into the prompt. Tool calling addresses a different gap: interacting with structured data and external APIs. When the data you need has a defined schema -- columns, fields, input-output relationships -- it can be represented as a function. Tool calling lets the LLM invoke that function.

The formal definition (adapted from IBM): tool calling allows autonomous systems to complete complex tasks by dynamically accessing and acting upon external resources. Two properties are central: the system completes a task, and it can rely on external resources to do so.

Though examples in this lecture use Python (because LLMs are heavily trained on Python code and read it fluently), tool calling is not tied to any specific programming language.

### 7.2 Anatomy of a Tool Definition

A tool is defined by its function API: the function name, its parameters with types, a natural language description of what it does, and the structure of its return value. The LLM never sees the implementation. It only sees the API signature and documentation.

For example, a `find_teddy_bear(latitude, longitude)` function might have the description "Finds the nearest available teddy bears given geographic coordinates" and return a structured object with fields like name, location, and availability. The implementation -- which API it calls, how it queries a database, how it handles errors -- is entirely on the backend, invisible to the model.

The description is critical. It is the primary signal the LLM uses to decide whether this tool is relevant to a given query and how to invoke it correctly.

### 7.3 The Three-Stage Tool Calling Mechanism

Tool calling proceeds in three stages, each involving a distinct role for the LLM or the system:

**Stage 1 -- Argument prediction (LLM call).** The user's query and the function API (without implementation) are placed in the prompt. The LLM's task is to determine which function to call and with what arguments. For example, given "find a teddy bear near me" and the `find_teddy_bear` API, the LLM extracts the user's coordinates from the conversation context and outputs a function call: `find_teddy_bear(latitude=37.43, longitude=-122.17)`.

**Stage 2 -- Function execution (system call, no LLM involved).** The system takes the LLM's predicted function call, executes it, and receives a structured result. This is ordinary code execution -- no model inference is involved.

**Stage 3 -- Response synthesis (LLM call).** The structured result is fed back to the LLM along with the full conversation history. The LLM translates the raw structured data (e.g., a JSON object) into a natural language response for the user: "I found a teddy bear called 'Cuddly' available at the Stanford Bookstore, 0.3 miles from you."

### 7.4 Training for Tool Calling: The SFT Approach

Training a model to use tools via supervised fine-tuning requires two sets of SFT pairs:

**Tool prediction pairs.** The input is the conversation history plus the function API. The target output is the correct function call with the correct arguments. Multiple examples vary the phrasing of queries, the location of grounding information (explicit coordinates vs. inferred from context), and the conversational setting (single-turn vs. multi-turn).

**Response synthesis pairs.** The input is the full conversation history including the tool call and its structured result. The target output is the natural language response. This pair teaches the model not just to paraphrase the JSON output but to connect it back to the original user intent, producing a coherent, contextual answer.

Diversity in training examples matters: the model should see varied phrasings, multi-turn conversations, and different ways the grounding information appears in the context.

### 7.5 Training for Tool Calling: The Prompt-Based Approach

Modern LLMs trained on large code corpora already understand function signatures, arguments, and return types. This raises the question: can we skip SFT entirely and rely on a well-crafted prompt to explain how to use a tool?

The prompt-based approach replaces SFT with a detailed explanation of the tool's behavior, written in natural language, that is prepended to the context at inference time. Writing this explanation well by hand is difficult, so practitioners use an iterative process:

1. Start with a draft explanation.
2. Assemble an evaluation set of (query, expected-tool-call) pairs (these can be repurposed SFT pairs).
3. Run the current explanation against the evaluation set and record successes and failures.
4. Feed the explanation, the evaluation results (wins and losses), and the failures to a powerful reasoning model, asking it to revise the explanation to fix the failures.
5. Repeat until evaluation performance converges.

This process produces a fixed prompt that is used at inference time alongside the function API. It avoids the cost of SFT retraining and works well because strong reasoning models are effective at writing precise, logically structured instructions. The recommendation is to never write the full explanation manually end-to-end -- let the model iterate on it using evaluation feedback.

### 7.6 Categories of Tool Use

Tools extend LLM capabilities across several categories:

**Informational tools** fetch data the model does not have: search engines for current news, weather APIs, stock market APIs, product databases. These directly address the knowledge cutoff problem in a structured, real-time way, complementing RAG's document-based approach.

**Computational tools** execute calculations. Rather than relying on the LLM's (imperfect) arithmetic reasoning, the model translates the query into code, executes it via a code interpreter tool, and reads the result. This is more reliable for precise computation.

**Action tools** perform operations on behalf of the user: sending emails, creating calendar events, placing orders, adjusting settings. These tools move the LLM from a passive question-answering system to an active agent that can affect the real world.

---

## 8. Tool Selection and Routing

### 8.1 The Scaling Problem

Real-world deployments expose the LLM to many tools simultaneously -- not just one teddy-bear finder but dozens or hundreds of APIs covering different capabilities. This creates two problems. First, the combined API documentation for all tools may exceed the context window or, even if it fits, trigger the needle-in-a-haystack degradation described earlier. Second, the model may confuse similar tools or fail to identify the correct one among too many options.

### 8.2 Two-Stage Tool Selection

The solution mirrors the two-stage retrieval architecture used in RAG. A paper from Google DeepMind introduces a **tool selector** (also called a **router**) that operates in two stages:

**Stage 1 -- Tool selection.** The user query and a lightweight list of all available tools (just names and one-line descriptions, not full APIs) are fed to an LLM. The model selects the subset of tools that might be relevant to the query.

**Stage 2 -- Full tool use.** Only the selected tools' full API documentation is inserted into the context alongside the query. The model then proceeds with the standard tool calling mechanism.

This approach keeps the per-query context small regardless of how many total tools are available, making the system scalable. Tool selection can also be implemented using RAG over tool descriptions -- embedding tool descriptions in a vector store and retrieving the most similar ones given a query -- though an LLM-based selector is also effective.

---

## 9. Model Context Protocol (MCP)

### 9.1 The Standardization Problem

Without a standard, every LLM provider defines its own format for specifying tools. Developers must re-implement the same tool for each provider, creating duplication and maintenance burden.

### 9.2 MCP Overview

The **Model Context Protocol (MCP)**, developed by Anthropic, standardizes how tools are exposed to models. It defines a common vocabulary:

- **MCP Server**: an instance that serves tools. Typically implemented by the party with domain expertise (e.g., a book provider runs the book recommendation MCP server).
- **Tools**: the function implementations available through the server.
- **Prompts**: template examples showing how to use the tools (usage hints for the model).
- **Resources**: external data sources (databases, user collections, catalogs) that tools can access to complete tasks.
- **MCP Client**: infrastructure within the LLM host that maintains a one-to-one connection with an MCP server.

For example, a poetry book recommendation scenario might involve Claude (the LLM host) connecting via an MCP client to a book provider's MCP server. The server exposes tools for finding books and recommending them, prompts demonstrating how to search by title or recommend by genre, and resources such as the user's personal collection or bestseller lists.

MCP enables a tool-once-use-anywhere model: a tool implemented as an MCP server can be consumed by any LLM host that speaks the protocol.

---

## 10. Agents

### 10.1 Definition

An agent is a system that autonomously pursues a goal and completes tasks on a user's behalf. The distinguishing feature compared to simple tool calling is the presence of iterative reasoning loops. A tool call is a single request-response cycle. An agent chains multiple tool calls together with intermediate reasoning, re-evaluating its progress after each step and deciding what to do next.

### 10.2 The ReAct Framework

ReAct (Reason + Act) is a foundational paper that formalizes the agentic loop. It decomposes complex tasks into cycles of three stages -- though the exact terminology varies across papers (think/observe/act, observe/plan/act, etc.), the structure is consistent:

**Observe** -- Interpret the current state of the world. Translate the user's request or the latest tool output into an actionable understanding. For example, "The user says their teddy bear is cold. This likely relates to room temperature, which is currently unknown."

**Plan** -- Decide what to do next. Identify which piece of missing information to obtain or which action to take. For example, "I need to determine the current room temperature. I have a `get_current_room_temperature` tool available."

**Act** -- Execute the plan by making a tool call (or producing a final response). For example, call `get_current_room_temperature()` and receive the result: 65 degrees Fahrenheit.

The agent then returns to the Observe stage with the new information: "65 degrees is colder than expected. I should increase the temperature." It Plans ("increase by 5 degrees"), Acts (`adjust_thermostat(temperature=70)`), Observes ("thermostat is now set to 70 degrees, task is complete"), and exits the loop with a natural language response to the user.

The key property is that the number of loop iterations is not fixed in advance. The agent continues until it determines the goal has been met or concludes it cannot make further progress.

### 10.3 Multi-Agent Systems

A single agent handles one reasoning loop with one set of tools. But complex real-world scenarios may require multiple specialized agents collaborating. A smart home, for instance, might have separate agents for thermostat control, energy management, and air quality. A user request like "make my home comfortable" could require coordination across all three.

### 10.4 Agent-to-Agent Protocol (A2A)

Google released the Agent-to-Agent (A2A) protocol to standardize inter-agent communication, analogous to how MCP standardizes tool exposure. Key elements of the specification include:

- **Skills**: each agent exposes a set of capabilities with descriptions so other agents know what it can do.
- **Execution lifecycle**: standardized status signals emitted during task execution, so coordinating agents can track progress.
- **Cancellation**: a defined process for one agent to instruct another to abort an ongoing task.

A2A and MCP address different layers of the stack: MCP standardizes how a single LLM interacts with tools, while A2A standardizes how multiple agents interact with each other.

---

## 11. Safety Considerations

### 11.1 New Attack Surface

Tool calling and agentic capabilities introduce risks that do not exist in a pure text-generation LLM. When a model can execute code, send emails, or write to public-facing systems, adversarial prompts can cause real-world harm.

**Data exfiltration** is a concrete example: if an agent has access to a tool that sends emails and also has access to the user's private data (passwords, documents), a malicious prompt could instruct the agent to email sensitive information to an external address.

The ToolSword paper catalogs a broader set of safety hazards specific to tool-augmented LLMs.

### 11.2 Mitigations

Defenses operate at two levels:

**Training-time defenses.** The SFT and reinforcement learning data mixtures include safety-oriented examples -- scenarios where the model must refuse harmful tool calls. The harmlessness component of RLHF training (covered in earlier lectures) extends naturally to tool use: the model learns that certain tool invocations are unsafe regardless of how they are phrased.

**Inference-time defenses.** A safety classifier monitors the conversation and the model's proposed actions, blocking tool calls that are flagged as potentially harmful before they execute. This provides a second layer of protection independent of the model's own training.

The **AgentSafetyBench** benchmark provides a comprehensive suite for evaluating LLM safety in agentic settings. Anthropic's recent disclosure of a large-scale cyberattack conducted via Claude's tool and agent capabilities underscores the urgency: attackers and defenders are both growing more sophisticated, and robust safety measures are not optional.

### 11.3 Error Compounding in Agentic Loops

Beyond adversarial attacks, agents face a reliability challenge inherent to their iterative structure. At each step in the reasoning loop -- argument prediction, tool selection, result interpretation -- the model can make a small error. Over multiple iterations, these errors compound. A wrong argument in step 2 produces an incorrect tool response, which leads to a flawed observation in step 3, which cascades forward. This compounding error rate is the primary reason fully autonomous agents have not yet reached the reliability needed for widespread unsupervised deployment.

---

## 12. Practical Advice for Building Tools and Agents

**Start small.** Begin with a single tool and a simple use case (e.g., "find the nearest teddy bear"). Verify that the tool definition, argument prediction, and response synthesis work correctly end-to-end before adding complexity.

**Start with the strongest model.** Use the most capable available model first to establish the ceiling of what is achievable. Once the pipeline works correctly, optimize for latency, cost, and smaller models.

**Inspect reasoning chains.** Agentic LLMs produce intermediate reasoning at each loop iteration. These chains are the primary debugging tool. When an agent fails, reading the observe-plan-act trace typically reveals where the reasoning diverged.

**Coding assistants as a real-world example.** AI-assisted coding is one of the most mature applications of agentic LLMs today. The agent can decompose a complex programming task, write code, run tests, interpret errors, and iterate. The caveat: generating code is cheap, but judging whether code is correct requires foundational knowledge. Taste and judgment remain the human's responsibility.

---

## Key Takeaways

1. **RAG** addresses the knowledge cutoff problem by retrieving relevant documents from a knowledge base and injecting them into the prompt, avoiding the pitfalls of retraining, context overflow, needle-in-a-haystack degradation, and excessive cost.

2. **Knowledge base construction** requires careful chunking (typically around 500 tokens with overlap) and embedding via encoder-only models like Sentence-BERT. Contextual chunking -- prepending LLM-generated context summaries to each chunk -- improves coherence at the cost of additional LLM calls, mitigated by prompt caching.

3. **Retrieval follows a two-stage pipeline**: candidate retrieval (bi-encoder semantic search, BM25 keyword matching, or a hybrid) reduces millions of chunks to hundreds, then reranking (cross-encoder scoring) produces the final top-K. HyDE can improve query-document embedding comparability by generating a synthetic document from the query before embedding.

4. **Retrieval quality** is measured by NDCG (ranking quality with position discounting), MRR (rank of first relevant result), Precision@K (fraction of retrieved documents that are relevant), and Recall@K (fraction of relevant documents that were retrieved). MTEB is the standard benchmark.

5. **Tool calling** extends LLMs beyond text generation to interact with structured data and external APIs. The three-stage mechanism -- argument prediction, function execution, response synthesis -- can be trained via SFT pairs or achieved through carefully iterated prompt-based explanations.

6. **Tool selection/routing** solves the scalability problem of too many tools by first selecting relevant tools from a lightweight list, then loading only those tools' full APIs into context.

7. **MCP** (Model Context Protocol) standardizes how tools are exposed to models, enabling tool-once-use-anywhere interoperability across LLM providers.

8. **Agents** add iterative reasoning loops (observe-plan-act) on top of tool calling, enabling multi-step task completion. The ReAct framework formalizes this decomposition. Multi-agent systems and the A2A protocol extend coordination across specialized agents.

9. **Safety** is non-negotiable in agentic systems. Training-time defenses (safety-oriented SFT/RLHF data), inference-time classifiers, and benchmarks like AgentSafetyBench are essential. Error compounding across reasoning loops remains the primary reliability bottleneck for autonomous agents.
