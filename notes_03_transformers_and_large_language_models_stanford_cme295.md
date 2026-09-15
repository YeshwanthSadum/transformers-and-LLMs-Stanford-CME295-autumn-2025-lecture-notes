# Lecture 3: Transformers & Large Language Models - Stanford CME 295

**Date:** October 10, 2025
**Instructors:** Afshine Amidi & Shervine Amidi
**Course:** CME 295 - Transformers and Large Language Models, Stanford

---

## 1. Recap: Three Categories of Transformer-Based Models

### 1.1 Encoder-Decoder, Encoder-Only, and Decoder-Only

The previous two lectures established the transformer architecture and the three families of models built on top of it. **Encoder-decoder** models (e.g., T5) retain both halves of the original transformer and handle text-in, text-out tasks such as translation and summarization. **Encoder-only** models (e.g., BERT) discard the decoder and produce rich contextual embeddings -- the CLS token embedding, for instance, is widely used for classification and sentiment analysis. **Decoder-only** models (e.g., GPT) discard the encoder and remove cross-attention, keeping only the masked self-attention and feed-forward layers. More than 90% of modern-day LLMs are decoder-only, making this the dominant paradigm going forward.

---

## 2. What Makes a Language Model "Large"

### 2.1 Defining an LLM

A **language model** assigns probabilities to sequences of tokens; specifically, it predicts the probability of the next token given the preceding context. A **large** language model is one that is scaled up along three axes:

1. **Model size** -- on the order of billions to hundreds of billions of parameters.
2. **Training data** -- hundreds of billions to tens of trillions of tokens.
3. **Compute** -- requires many GPUs for both training and inference.

The term "LLM" has become well-established only in recent years. In 2018--2019, the terminology was loose; some people included BERT under the LLM umbrella. Under the modern consensus definition, BERT does not qualify because it is encoder-only and does not generate text. An LLM must be a generative, text-to-text model that is large in parameters, data, and compute.

### 2.2 Architecture of Modern LLMs

Modern LLMs are decoder-only. Each block consists of masked self-attention, a feed-forward neural network, and layer normalization (with residual connections). Notable examples include GPT (OpenAI), Llama (Meta), Gemma (Google), DeepSeek, Mistral, and Qwen.

---

## 3. Mixture of Experts (MoE)

### 3.1 Motivation: Not All Parameters Need to Activate

Models with hundreds of billions of parameters are expensive to run. The key insight behind Mixture of Experts is that not every parameter needs to participate in every forward pass. Consider a room containing a mathematician, a physicist, a chemist, and a historian. If you have a math question, you would consult the mathematician -- not everyone. MoE applies this logic to neural network components: given an input, only a subset of "expert" sub-networks should be activated.

### 3.2 Formulation

Let there be $n$ experts $E_1, E_2, \dots, E_n$, each a separate sub-network, and a gate (also called a **router**) $G$. Given input $x$, the output is:

$$\hat{y} = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)$$

where $G(x)_i$ is the weight (probability) the gate assigns to expert $i$, and $E_i(x)$ is the output of expert $i$. Both the gate and the experts are trained jointly via standard backpropagation.

### 3.3 Dense vs. Sparse MoE

**Dense MoE** places no constraint on how many experts are active. The gate produces a full probability distribution over all experts, weighting each output accordingly. Every expert contributes to every prediction, though some more than others.

**Sparse MoE** selects only the top-$K$ experts (typically $K = 1$ or $K = 2$) and zeroes out the rest:

$$\hat{y} = \sum_{i \in \text{top-}K} G(x)_i \cdot E_i(x)$$

Sparse MoE is the variant used in modern LLMs because it reduces the number of **floating-point operations (FLOPs)** per forward pass. FLOPs measure computational cost -- the count of additions and multiplications in a pass. With sparse MoE, you can scale total model parameters (increasing model *capacity*) without proportionally increasing inference-time compute, because only the *active parameters* (those belonging to the selected experts) participate in each forward pass.

### 3.4 Where Experts Live: The Feed-Forward Network

In a decoder block there are three main components: masked self-attention, the feed-forward network (FFN), and normalization. The FFN is where MoE replaces a single network with multiple expert networks, because the FFN is by far the most parameter-heavy component.

The FFN projects from $d_{\text{model}}$ to a hidden dimension $d_{\text{ff}}$ and back. Its parameter count is on the order of $2 \cdot d_{\text{model}} \cdot d_{\text{ff}}$, where $d_{\text{ff}}$ is typically thousands to tens of thousands. The attention layer, by contrast, involves projection matrices of dimension $d_{\text{model}} \times d_k$ (or $d_v$), where $d_k$ is typically in the hundreds -- much smaller.

Each expert is itself a feed-forward neural network. The routing decision happens at the **token level**: after the self-attention layer produces a contextualized representation of a token, that representation is fed to the gate, which selects the top-$K$ expert(s) for that specific token. Different tokens in the same sequence can be routed to different experts. The gate is **layer-specific** -- each decoder layer has its own independent, trainable gate -- and expert weights are not shared across layers.

### 3.5 Routing Collapse and the Auxiliary Load-Balancing Loss

A practical challenge during training is **routing collapse**: the gate learns to route tokens to the same one or two experts, leaving others permanently idle. To mitigate this, an auxiliary loss term is added:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot n \cdot \sum_{i=1}^{n} f_i \cdot P_i$$

where:
- $\alpha$ is a hyperparameter controlling the strength of the regularization,
- $n$ is the number of experts,
- $f_i$ is the fraction of tokens routed to expert $i$ in a given batch,
- $P_i$ is the average routing probability for expert $i$ across the batch.

This loss pushes both $f_i$ and $P_i$ toward uniform distributions, encouraging all experts to receive a roughly equal share of tokens. An additional technique called **noisy gating** adds random noise to the gate's logits before the top-$K$ selection, giving otherwise-neglected experts a chance to be activated -- conceptually similar to dropout as a regularization tool.

### 3.6 Scaling Properties

MoE-based models can have total parameter counts in the trillions (e.g., the Switch Transformer scaled to over one trillion parameters) while keeping active parameters per token much smaller. These models are also more **sample-efficient**: they reach a given performance level faster during training compared to dense models of equivalent total parameter count, because each expert can specialize.

---

## 4. Next Token Prediction: Decoding Strategies

### 4.1 The Output Probability Distribution

At each generation step, the decoder produces a contextualized embedding for the current position. This embedding passes through a linear layer that projects from $d_{\text{model}}$ to the vocabulary size $|V|$, followed by a softmax to produce a probability distribution over all tokens:

$$P(\text{token}_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$

where $z_i$ is the logit for token $i$ and $T$ is the **temperature** hyperparameter (discussed in Section 4.5).

### 4.2 Greedy Decoding

The simplest strategy: always pick the token with the highest probability. This is deterministic -- the same input always yields the same output. Two drawbacks:

1. **Lack of diversity.** The output never varies across runs.
2. **Local optimality.** Choosing the locally highest-probability token at each step does not guarantee the globally highest-probability *sequence*. A slightly less probable token at step $t$ might unlock a much higher-probability continuation at step $t+1$.

### 4.3 Beam Search

Beam search addresses the local-optimality problem by maintaining the $K$ most probable partial sequences (where $K$ is the **beam width**). At each step, every beam is extended by all possible next tokens, and only the top-$K$ resulting sequences (ranked by cumulative log-probability) are retained.

The sequence score is:

$$\log P(\text{sequence}) = \sum_{t=1}^{T} \log P(\text{token}_t \mid \text{token}_{<t})$$

Because each term is negative (probabilities are between 0 and 1), longer sequences accumulate lower scores. Beam search therefore includes a **length normalization** factor (e.g., dividing by $T^\alpha$ for some $\alpha$) to avoid systematically favoring shorter outputs.

Beam search is more globally optimal than greedy decoding but still deterministic and computationally heavier. It is used in settings like machine translation where high-likelihood outputs are desirable, but it is not the standard for general LLM generation because it lacks diversity and creativity.

### 4.4 Sampling-Based Decoding

Instead of deterministically selecting the best token, **sampling** draws the next token randomly according to the output probability distribution. Higher-probability tokens are more likely to be drawn, but lower-probability tokens have a nonzero chance. This introduces diversity across runs.

Two refinements restrict the sampling pool:

- **Top-$K$ sampling:** Only the $K$ highest-probability tokens are eligible; probabilities are renormalized over this subset.
- **Top-$P$ (nucleus) sampling:** The smallest set of tokens whose cumulative probability exceeds a threshold $P$ is kept; probabilities are renormalized over this set.

Both methods prevent the model from sampling very unlikely tokens (which would produce incoherent output) while preserving creative variation.

### 4.5 Temperature

Temperature $T$ appears in the softmax and controls the sharpness of the output distribution:

$$P(\text{token}_i) = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$

**Low temperature ($T \to 0$):** The distribution becomes spiky. To see why, factor out the maximum logit $z_k$:

$$P(\text{token}_i) = \frac{\exp\bigl((z_i - z_k)/T\bigr)}{\sum_j \exp\bigl((z_j - z_k)/T\bigr)}$$

For $i = k$, the numerator exponent is $0/T = 0$, so the numerator is 1. For $i \neq k$, $(z_i - z_k) < 0$, so dividing by a small $T$ drives the exponent to $-\infty$ and the term to 0. The result: nearly all probability mass concentrates on the single highest-logit token. Low temperature therefore behaves like greedy decoding and produces deterministic, "safe" output.

**High temperature ($T \to \infty$):** Every exponent $(z_j - z_k)/T \to 0$, so $\exp(\cdot) \to 1$ for all tokens. The distribution flattens to uniform ($1/|V|$), and sampling becomes essentially random. High temperature produces diverse, creative -- but potentially incoherent -- output.

In practice, a strictly positive temperature means every run can produce a different output. Setting $T = 0$ would theoretically make output fully deterministic, but in practice, hardware-level nondeterminism (floating-point reduction order on GPUs) can still cause slight variation. A recent paper, "Defeating Nondeterminism in LLM Inference," explores this phenomenon in detail.

### 4.6 Guided Decoding

When the output must conform to a specific format (e.g., valid JSON), **guided decoding** constrains the set of allowed next tokens at each generation step using formal grammar rules (finite state machines, context-free grammars). For example, after generating an opening brace `{`, only a quoted property name is a valid next token. When multiple tokens are permissible, standard sampling strategies apply among them. This eliminates the need for naive retry loops ("generate, check validity, regenerate if invalid").

---

## 5. Prompting Strategies

### 5.1 Context Length

The number of input tokens is called the **context length** (also context size, window size). Modern LLMs support context lengths on the order of tens of thousands to millions of tokens. Models like Gemini advertise context windows in the million-token range.

However, longer context does not guarantee better performance. A recent phenomenon called **context rot** (documented in a 2025 paper) shows that as context length grows, the model's ability to retrieve specific information buried in the text ("needle in a haystack") degrades, especially in the presence of distractors. This motivates targeting the *right* context rather than simply providing *more* context, which is a core rationale behind retrieval-augmented generation.

The context length is directly tied to the self-attention computation -- each token attends to all preceding tokens within the window. Efficiency techniques (from Lecture 2) manage the $O(n^2)$ complexity as context length grows.

### 5.2 Prompt Structure

There is no formal theory of prompt design, but a useful mental model decomposes any prompt into four parts:

1. **Context** -- sets up the scenario, background information, system identity.
2. **Instructions** -- the task the model should perform.
3. **Input** -- the specific data or question to process.
4. **Constraints** -- output format requirements, safety guardrails, style rules.

For example, a chatbot prompt might include: *Context:* "You are ChatGPT, the date is October 10, 2025." *Instructions:* the user's query. *Input:* any attached documents. *Constraints:* "Do not generate harmful content."

### 5.3 In-Context Learning: Zero-Shot and Few-Shot

**In-context learning** refers to steering an LLM's behavior through the prompt alone, without updating any model weights.

- **Zero-shot:** The prompt contains only the task description and input. No examples are provided.
- **Few-shot:** The prompt includes several input-output example pairs before the actual query. The model "learns" the task pattern from these examples.

Few-shot prompting generally improves performance because the examples demonstrate the expected format and reasoning pattern. However, it increases token usage (more compute) and requires curating good examples. An emerging trend with more capable reasoning models is that well-crafted zero-shot instructions can match or even exceed few-shot performance, because providing fixed examples can constrain the model to a narrow distribution, whereas natural-language instructions let the model leverage its reasoning abilities more flexibly. The "Plan and Solve" paper explores this direction.

### 5.4 Chain of Thought (CoT)

**Chain of thought** prompting forces the model to produce intermediate reasoning steps before arriving at a final answer. Rather than jumping directly to a conclusion, the model generates a "chain" of logical deductions.

Why this works: the transformer generates tokens autoregressively, and each generated token becomes part of the context for subsequent tokens. By emitting reasoning steps, the model effectively gives itself "scratch space" to decompose complex problems. Empirically, CoT leads to measurable improvements on arithmetic, logic, and multi-step reasoning benchmarks.

CoT also serves as a **debugging tool**. When a model produces an incorrect answer, examining the chain of thought reveals where the reasoning went wrong (e.g., the model stated an incorrect date), enabling targeted fixes to the prompt or context.

The trade-off is increased output length (more tokens generated), which means higher latency and cost. In practice, this is typically acceptable given the quality improvement.

### 5.5 Self-Consistency

**Self-consistency** builds on chain of thought by sampling the model's response multiple times (in parallel, not sequentially) and applying **majority voting** on the final answers.

The procedure:
1. Issue the same prompt $N$ times, each with stochastic sampling (positive temperature).
2. Each response includes a chain of thought followed by a final answer.
3. Extract the final answer from each response (using regex, structured output constraints, or even another LLM).
4. The answer that appears most frequently across the $N$ samples is selected.

Because the $N$ samples are generated in parallel, latency is roughly equal to the slowest single generation, not $N$ times a single generation. Self-consistency improves robustness on mathematical and arithmetic benchmarks where chain-of-thought alone can still produce errors.

---

## 6. Inference Optimization

Inference optimization techniques divide into **exact** methods (producing identical results to the unoptimized computation, just faster) and **approximate** methods (producing slightly different results at lower cost).

### 6.1 KV Cache

During autoregressive generation, each new token must attend to all previous tokens. Naively, this means recomputing the key ($K$) and value ($V$) projections for every prior token at every step. The **KV cache** stores these projections after they are computed once and reuses them at subsequent steps.

Only keys and values are cached -- not queries, because the query for a past token is irrelevant to the current token's attention computation. The KV cache is strictly an inference-time optimization; during training, teacher forcing processes all positions simultaneously, so caching is unnecessary.

### 6.2 Grouped Query Attention (GQA)

Recall from Lecture 2 that multi-head attention maintains $H$ independent sets of query, key, and value projection matrices. **Grouped query attention** reduces the number of key and value heads by sharing them across groups of query heads. The two extremes are:

- $H$ key/value heads = full multi-head attention (no sharing).
- 1 key/value head = multi-query attention (maximum sharing).

GQA with an intermediate group size is the common choice in modern LLMs. It directly reduces the size of the KV cache by a factor proportional to the grouping ratio.

### 6.3 Paged Attention

When serving multiple users, a naive memory allocation strategy reserves space for the full maximum context length per request. Since generation length is unknown in advance (decoding stops at the EOS token), this leads to significant **memory waste** through:

- **Internal fragmentation:** Reserved space for tokens not yet generated (and possibly never generated).
- **External fragmentation:** Gaps left by the memory allocator between allocated blocks.

**Paged Attention** (introduced in the vLLM paper) borrows ideas from operating system virtual memory. Instead of allocating one contiguous block per request, it breaks the KV cache into fixed-size blocks (e.g., 16 tokens each). A page table maps logical token positions to physical memory blocks. Blocks are allocated on demand as generation proceeds, dramatically reducing fragmentation and allowing more concurrent requests to be served.

### 6.4 Multi-Latent Attention (MLA)

DeepSeek introduced **multi-latent attention** to further compress the KV cache. In standard multi-head attention, each of the $H$ heads has its own key and value projection, resulting in $H$ key vectors and $H$ value vectors per token per layer -- all of which must be cached.

MLA factorizes the projection through a low-dimensional bottleneck:

1. A **shared compression matrix** $W_{\text{down}}$ projects the token representation from $d_{\text{model}}$ to a much smaller latent dimension $d_c$ (where $d_c \ll d_k$).
2. Separate **decompression matrices** $W_{\text{up}}^K$ and $W_{\text{up}}^V$ expand back to key and value spaces, respectively.

The compression matrix is shared across all heads *and* across keys and values. This means only a **single compressed vector per token per layer** needs to be stored in the cache, rather than $2H$ full-dimensional vectors. The latent dimension $d_c$ is a fixed design choice.

Beyond memory savings, the DeepSeek-V2 paper reported performance improvements, attributed to a regularization effect: forcing keys and values through a shared bottleneck encourages more generalizable representations.

### 6.5 Speculative Decoding

**Speculative decoding** exploits the observation that at inference time, LLM generation is **memory-bound**, not compute-bound. A single forward pass over many tokens costs roughly the same as generating one token at a time, because the bottleneck is loading model weights from memory, not performing arithmetic.

The procedure:
1. A small, fast **draft model** generates $K$ candidate tokens autoregressively (e.g., "is", "cute", "and", "smart").
2. All $K$ draft tokens are fed as input to the large **target model** in a single forward pass, obtaining the target model's probability distribution at each position.
3. An **acceptance/rejection scheme** compares draft and target distributions at each position:
   - If $Q_{\text{target}}(\text{token}_t) \geq P_{\text{draft}}(\text{token}_t)$, the token is accepted.
   - Otherwise, the token is accepted with probability $Q_{\text{target}} / P_{\text{draft}}$ and rejected otherwise.
4. If all tokens are accepted, the model has advanced by $K$ tokens in the time of roughly one forward pass. If a token is rejected, generation resumes from that position with an adjusted distribution.

A bonus: the target model's forward pass also produces the probability distribution for the position *after* the last draft token, which can be sampled for free.

The mathematical guarantee is that this procedure produces the **exact same distribution** as sampling directly from the target model. This follows from the law of total probability -- the proof is a short derivation in the original paper. The technique is a form of **rejection sampling**.

### 6.6 Multi-Token Prediction

**Multi-token prediction** embeds the draft model *inside* the target model itself. Instead of a separate smaller model, the architecture attaches **multiple prediction heads** to the final decoder representation. During training, the objective changes from next-single-token prediction to next-$K$-token prediction simultaneously.

At inference time:
- The additional heads act as the draft model, producing $K$ candidate tokens.
- The primary (first) head acts as the target model, verifying the candidates.
- An acceptance/rejection step (done greedily in the original paper) determines how many tokens to keep.

This approach eliminates the need to maintain a separate draft model and naturally encourages the shared representations to be useful for predicting multiple future tokens. The trade-off is a modified training objective, which means the exact distribution-matching property of speculative decoding does not hold in the same form, though a greedy variant achieves strong results in practice.

---

## Key Takeaways

1. **LLMs are decoder-only transformers** scaled up across three dimensions: parameters (billions), training data (trillions of tokens), and compute (many GPUs). BERT and other encoder-only models are not LLMs under the modern definition.

2. **Mixture of Experts** allows scaling total parameters without proportionally increasing inference cost. Sparse MoE activates only top-$K$ experts (typically $K = 1$ or $2$) per token, with routing handled by a trainable gate at the FFN layer. Auxiliary losses and noisy gating prevent routing collapse.

3. **Decoding strategies** trade off between quality, diversity, and cost. Greedy decoding is deterministic but locally optimal. Beam search is more globally optimal but still deterministic and expensive. Sampling with top-$K$/top-$P$ filtering provides diversity. Temperature controls the sharpness of the distribution: low temperature concentrates probability on the most likely token; high temperature flattens toward uniform.

4. **Guided decoding** constrains token generation to valid outputs (e.g., JSON) using formal grammars, avoiding retry loops.

5. **Prompting strategies** can dramatically change LLM behavior without weight updates. Few-shot examples steer the model toward a task, chain of thought forces explicit reasoning (improving accuracy and debuggability), and self-consistency uses majority voting over parallel samples for robustness.

6. **Inference optimizations** span exact and approximate methods. KV caching avoids redundant key/value computation. GQA reduces cache size by sharing key/value heads. Paged attention borrows virtual memory concepts to reduce memory fragmentation. Multi-latent attention compresses the cache through a shared low-rank bottleneck. Speculative decoding uses a small draft model to propose tokens verified by the target model in a single pass, provably matching the target distribution. Multi-token prediction internalizes the draft model by training with a multi-token objective.
