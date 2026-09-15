# Lecture 4: LLM Training - Stanford CME 295

**Date:** October 17, 2025
**Instructors:** Afshine Amidi & Shervine Amidi (Adjunct Lecturers, Stanford)
**Course:** CME 295 - Transformers and Large Language Models, Autumn 2025

---

## 1. Transfer Learning

Traditionally in machine learning, each task demanded its own model trained from scratch. If you wanted spam detection, you trained a spam detector. If you wanted sentiment analysis, you trained a sentiment analyzer. These models shared no knowledge, even though both fundamentally require understanding text.

Transfer learning breaks this pattern. The core insight is that language understanding is a shared skill: the ability to parse syntax, track meaning across sentences, and reason about context is useful for spam detection, summarization, translation, and virtually every other NLP task. Rather than learning these skills independently each time, we can learn them once and reuse them.

The transfer learning paradigm for LLMs has two stages:

1. **Pre-training:** Train a large model on vast amounts of data so it learns the general structure of language and code. This stage is task-agnostic -- the model is not solving any particular downstream problem, just learning to predict text.
2. **Fine-tuning:** Take the pre-trained model and adapt its weights to a specific task (spam detection, sentiment analysis, instruction following, etc.) using a much smaller, task-specific dataset.

This means that for every new task, you do not start from scratch. You start from a model that already understands language and simply steer it toward your use case. This paradigm is the foundation on which all modern LLMs are built.

---

## 2. Pre-Training

Pre-training is by far the most expensive stage of LLM training -- in compute, data, time, and cost. Its purpose is to teach the model the structure of language and the breadth of human knowledge by training it to predict the next token on enormous corpora of text.

### 2.1 The Objective

The pre-training objective is **next-token prediction**. The model, which is almost always a decoder-only transformer (over 90% of modern LLMs), takes a sequence of tokens as input and predicts the next token at each position. This is repeated over trillions of tokens until the model develops a robust internal representation of language, code, and factual knowledge.

### 2.2 Pre-Training Data

The data used for pre-training is essentially "everything you can find." Key sources include:

- **Common Crawl:** A massive, continuously updated web crawl containing roughly 3 billion pages per month. This forms the backbone of most pre-training datasets.
- **Wikipedia:** High-quality encyclopedic text across many languages.
- **Social media and forums:** Reddit conversations, discussion boards, and similar sources that capture informal language, dialogue, and argumentation.
- **Code repositories:** GitHub, Stack Overflow, and programming forums provide the data needed for code understanding and generation.

The scale of these datasets is measured in tokens:

| Model | Pre-Training Tokens |
|---|---|
| GPT-3 | 300 billion |
| LLaMA 3 | 15 trillion |

The order of magnitude to remember is **hundreds of billions to tens of trillions of tokens**.

### 2.3 Pre-Training Challenges

Pre-training carries several significant challenges:

- **Cost:** Training runs cost millions to hundreds of millions of dollars. This is not an exaggeration -- every major model release includes a cost figure in this range.
- **Time:** Pre-training runs take weeks to months on thousands of GPUs.
- **Environmental impact:** The energy consumption of these runs has prompted the community to report ecological costs alongside performance metrics.
- **Knowledge cutoff:** The pre-trained model's knowledge is frozen at the date its training data was collected. It has no way to know about events that occurred after this date. Model cards always report a knowledge cutoff date (e.g., GPT-5's cutoff is listed as September 30, 2025).
- **Knowledge editing:** Injecting new knowledge into an already-trained model without degrading performance in other domains remains an open and difficult problem.
- **Plagiarism risk:** Because the model is trained to predict text it has seen, there is always a risk of it reproducing training data verbatim.

---

## 3. FLOPS vs. FLOP/s

Two notations appear constantly in discussions of training compute, and they are easy to confuse because they are spelled almost identically.

### 3.1 FLOPS (Floating Point Operations)

FLOPS (sometimes written FLOPs) is a **unit of total compute**. It counts the total number of arithmetic operations on floating-point numbers required to complete a task (such as training a model). Training a modern LLM is on the order of $10^{25}$ FLOPS.

The total FLOPS for a training run is roughly proportional to the product of the number of model parameters and the number of training tokens:

$$\text{FLOPS} \approx O(N \times D)$$

where $N$ is the number of parameters and $D$ is the number of training tokens. The exact formula depends on the architecture (e.g., sparse MoE models require fewer FLOPS than dense models of the same parameter count because only a subset of parameters is activated per token), but this approximation captures the key relationship.

### 3.2 FLOP/s (Floating Point Operations Per Second)

FLOP/s is a **measure of hardware speed** -- how many floating-point operations a device can execute per second. GPU spec sheets always list FLOP/s for different numerical precisions (FP64, FP32, FP16, etc.). For example, the NVIDIA H100 has different FLOP/s ratings depending on the data type used for computation.

The convention is that FLOP/s is written in all capitals (FLOPS) whereas total operations (FLOPS/FLOPs) may use a lowercase 's'. In practice, authors sometimes use one notation for the other, so you must rely on context to disambiguate.

---

## 4. Scaling Laws and the Chinchilla Result

### 4.1 The Original Scaling Laws (Kaplan et al., 2020)

The paper "Scaling Laws for Neural Language Models" (2020) systematically varied model size, dataset size, and compute budget, then measured how next-token prediction loss changed. The findings were:

1. **More compute leads to lower loss.** Performance improves predictably as you add compute.
2. **More data leads to lower loss.** Larger training sets produce better models.
3. **Larger models lead to lower loss.** Increasing the number of parameters improves performance.
4. **Larger models are more sample-efficient.** For a fixed number of tokens processed, a larger model achieves lower loss than a smaller one. In other words, bigger models extract more learning per token.

These findings drove a period (roughly 2019-2024) where the dominant strategy was to build ever-larger models, since the scaling curves showed no sign of saturating.

### 4.2 Compute-Optimal Training (Chinchilla)

Given a fixed compute budget, there is a trade-off between model size and dataset size. You can build a large model trained on fewer tokens, or a smaller model trained on more tokens -- the total FLOPS is roughly the same in both cases.

The Chinchilla research addressed the question: **for a given compute budget, what is the optimal balance between model size and training tokens?**

By fixing a compute budget (represented by curves of the same color on the loss landscape) and training models of different sizes with correspondingly different amounts of data, the researchers found a consistent sweet spot. The key finding:

$$D_{\text{optimal}} \approx 20 \times N$$

where $D_{\text{optimal}}$ is the optimal number of training tokens and $N$ is the number of model parameters. In other words, the training dataset should be roughly 20 times the parameter count for compute-optimal training.

By this standard, GPT-3 (175 billion parameters, 300 billion tokens) was significantly **undertrained** -- it had a token-to-parameter ratio of less than 2, far below the recommended 20. This insight shifted the field's strategy: instead of always making models bigger, teams began investing more heavily in training data quality and quantity.

It is worth noting that different teams reproduce this analysis for their own architectures and setups. For example, the LLaMA 3 paper includes a dedicated section where the authors ran small-scale experiments to determine the optimal parameter-token ratio for their specific architecture, then extrapolated to set the final model size. The Chinchilla ratio of 20:1 is a useful guideline, but the exact optimum varies by architecture, data quality, and other factors.

---

## 5. GPU Training and Memory Constraints

### 5.1 Why GPUs

LLMs are decoder-only transformer models, and the dominant operation inside a transformer is matrix multiplication. GPUs are hardware specifically optimized for massively parallel matrix operations, making them the natural choice for training. (Google uses their own custom hardware called TPUs, but virtually all non-Google models are trained on GPUs.)

### 5.2 The Training Loop and Memory Requirements

Training an LLM consists of three stages repeated over the dataset:

1. **Forward pass:** Data flows through the network, producing predictions. At each layer, intermediate values called **activations** are computed and must be stored in memory (they are needed later for the backward pass). The memory consumed by activations depends on model size, batch size, and context length. Context length is particularly expensive because self-attention has $O(n^2)$ complexity in sequence length $n$.

2. **Backward pass:** The gradient of the loss with respect to each parameter is computed via backpropagation. This requires accessing the stored activations and produces gradient tensors that also consume memory.

3. **Weight update:** An optimizer (typically Adam) applies the gradients to update the model weights. Adam maintains two additional quantities per parameter -- the first moment (running mean of gradients) and the second moment (running mean of squared gradients) -- which must also be stored in memory.

### 5.3 The Memory Wall

GPU memory is limited. An NVIDIA H100 has 80 GB of GPU memory (HBM). Into this 80 GB, you must fit:

- The model parameters
- The activations from the forward pass
- The gradients from the backward pass
- The optimizer states (first and second moments for Adam)

For a model with hundreds of billions of parameters, this is not possible on a single GPU. The solution is to distribute the workload across many GPUs.

---

## 6. GPU Parallelism Strategies

### 6.1 Data Parallelism (DP)

Data parallelism distributes the training data across multiple GPUs while keeping a full copy of the model on each GPU.

**How it works:**
1. The training batch is divided into sub-batches, one per GPU.
2. Each GPU holds a complete copy of the model.
3. Each GPU independently performs the forward and backward pass on its sub-batch.
4. The gradients from all GPUs are averaged (via inter-GPU communication) and applied to update the weights.

**Benefit:** Reduces the memory burden associated with batch size, since each GPU only processes a fraction of the total batch.

**Limitations:**
- Each GPU must still hold a full copy of the model. If the model itself does not fit in one GPU's memory, data parallelism alone is insufficient.
- Communication costs for gradient aggregation increase with the number of GPUs, slowing training.

### 6.2 ZeRO (Zero Redundancy Optimizer)

Standard data parallelism is wasteful: every GPU stores identical copies of the model parameters, gradients, and optimizer states. ZeRO eliminates this redundancy by **partitioning** (sharding) these quantities across GPUs instead of replicating them.

ZeRO has three progressive stages:

| Stage | What is partitioned | Memory savings |
|---|---|---|
| ZeRO-1 | Optimizer states only | Large |
| ZeRO-2 | Optimizer states + gradients | Larger |
| ZeRO-3 | Optimizer states + gradients + parameters | Maximum (no redundancy) |

**Trade-off:** Each additional stage saves more memory but increases communication costs, because GPUs must now exchange data that they previously held locally. The choice of ZeRO stage depends on how large the model is relative to per-GPU memory and how sensitive the training pipeline is to communication overhead.

### 6.3 Model Parallelism

While data parallelism distributes data across GPUs, model parallelism distributes the model itself. Several variants exist:

- **Tensor parallelism:** Large matrix multiplications within a single layer are split across GPUs. Each GPU computes a portion of the result, reducing the per-GPU memory needed for that operation.
- **Pipeline parallelism:** The model's layers are divided into contiguous groups, each assigned to a different GPU. GPU 1 handles layers 1-3, GPU 2 handles layers 4-6, and so on. During the forward pass, activations flow from one GPU to the next.
- **Expert parallelism:** For Mixture-of-Experts models (covered in Lecture 3), different experts are placed on different GPUs. Since only a subset of experts is activated per token, this maps naturally to a distributed setup.

The details of each variant are less important than the conceptual understanding: data parallelism splits the data, model parallelism splits the model, and in practice large-scale training uses both simultaneously.

---

## 7. Flash Attention

Flash attention is a technique developed at Stanford (Dao et al., 2022) that makes the self-attention computation significantly faster and more memory-efficient without changing the result -- it is an **exact** computation, not an approximation.

### 7.1 GPU Memory Hierarchy

To understand flash attention, you need to know that a GPU has two types of memory:

| Memory | Capacity | Speed |
|---|---|---|
| **HBM** (High Bandwidth Memory) | Tens of GB (e.g., 80 GB on H100) | A few TB/s |
| **SRAM** (on-chip) | Tens of MB | Tens of TB/s (~10x faster than HBM) |

HBM is large but slow. SRAM is tiny but extremely fast (roughly 10x the bandwidth of HBM). The key bottleneck in attention is not arithmetic -- GPUs are fast at multiplying matrices -- but **data movement** between HBM and SRAM.

### 7.2 The Problem with Standard Attention

The self-attention formula is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q, K, V \in \mathbb{R}^{n \times d}$ are the query, key, and value matrices, $n$ is the sequence length, and $d_k$ is the head dimension.

In a naive implementation, the computation proceeds as:

1. Load $Q$ and $K$ from HBM, compute $S = QK^\top$, write $S$ back to HBM.
2. Load $S$ from HBM, compute $P = \text{softmax}(S)$, write $P$ back to HBM.
3. Load $P$ and $V$ from HBM, compute $O = PV$, write $O$ back to HBM.

Each step involves a round trip to the large but slow HBM. This repeated reading and writing becomes the dominant cost -- the computation is **memory-bound**, not compute-bound.

### 7.3 Tiling: The Core Idea

Flash attention minimizes HBM reads and writes by processing attention in small **tiles** (blocks) that fit entirely in SRAM. Instead of materializing the full $n \times n$ attention matrix in HBM, it:

1. Loads small blocks of $Q$, $K$, and $V$ from HBM into SRAM.
2. Computes the full attention output for those blocks entirely within SRAM.
3. Writes only the final result back to HBM.

The mathematical trick that makes this possible is that softmax can be decomposed across blocks. Although softmax normalizes across an entire row (each row must sum to 1), you do not need the full row computed at once. If a row is split into sub-blocks $S_1, S_2, \ldots, S_n$, the softmax of the full row equals the block-wise softmax of each sub-block, corrected by a scaling factor. This scaling factor is computed iteratively as each new block is processed, using the running maximum and sum of exponentials.

The algorithm iterates over blocks of keys/values for each block of queries, accumulating the output and updating the scaling factors on the fly. The result is mathematically identical to standard attention.

### 7.4 Recomputation: Trading Compute for Memory

The second idea in the flash attention paper addresses the backward pass. Normally, activations from the forward pass (specifically, the attention matrix $P$) are saved in memory so they can be reused during gradient computation. These activations consume substantial memory.

Because flash attention makes the forward computation so fast, the paper proposes **recomputation**: instead of storing the attention activations, discard them after the forward pass and recompute them on the fly during the backward pass using the same tiled algorithm.

This approach performs more total floating-point operations (you compute the attention twice), but it achieves:

- **Fewer HBM reads/writes:** Roughly a 10x reduction (e.g., from 40.3 GB of HBM accesses down to approximately 4 GB in the benchmarks reported).
- **Faster wall-clock time:** Despite more total operations, the reduced data movement makes the overall computation faster.
- **Lower memory usage:** No need to store the $n \times n$ attention matrix.

This is a remarkable result: recomputation usually trades speed for memory, but here you get improvements in both dimensions simultaneously because the bottleneck was data movement, not arithmetic.

### 7.5 Subsequent Versions

Flash Attention 2 and Flash Attention 3 are subsequent iterations that adapt the core tiling and recomputation ideas to newer GPU architectures, exploiting hardware-specific features of each new generation. The fundamental principles remain the same.

---

## 8. Quantization and Mixed Precision Training

### 8.1 Floating-Point Representations

Every weight, activation, and gradient in a neural network is stored as a floating-point number. A floating-point number is encoded as a sequence of bits divided into three fields:

- **Sign bit:** 1 bit indicating positive or negative.
- **Exponent bits:** Determine the range (how large or small the number can be).
- **Mantissa bits:** Determine the precision (how many significant digits after the decimal point).

Common representations and their bit budgets:

| Format | Total Bits | Exponent | Mantissa | Memory per Value |
|---|---|---|---|---|
| FP64 (double precision) | 64 | 11 | 52 | 8 bytes |
| FP32 (single precision) | 32 | 8 | 23 | 4 bytes |
| FP16 (half precision) | 16 | 5 | 10 | 2 bytes |
| BF16 (brain float 16) | 16 | 8 | 7 | 2 bytes |

FP16 uses half the memory of FP32 but is less precise (fewer mantissa bits mean less granularity). BF16 keeps the same exponent range as FP32 (8 exponent bits) but sacrifices mantissa precision, making it well-suited for deep learning where range matters more than fine-grained precision.

### 8.2 Quantization

Quantization is the process of converting numbers from a higher-precision format to a lower-precision one. The motivation is straightforward: if you can represent weights in 16 bits instead of 32 without meaningfully degrading model quality, you halve the memory footprint and can also compute faster (GPUs achieve higher FLOP/s at lower precisions).

GPU spec sheets reflect this directly. On the H100, FP64 operations run at 34 teraFLOP/s, while lower-precision formats achieve substantially higher throughput. Dropping from FP64 to FP32 roughly doubles compute speed, and dropping further to FP16 or BF16 doubles it again.

There are several quantization techniques (e.g., zero-point quantization and absmax quantization) that differ in how they map the continuous range of values to discrete quantized levels. These are not covered in depth here but are worth noting as practical tools.

### 8.3 Mixed Precision Training

Mixed precision training combines different numerical precisions for different parts of the training loop to get the benefits of lower precision (speed and memory) without the drawbacks (accumulated rounding errors in the weights).

The strategy:

1. **Store the master copy of weights in FP32** (high precision). This is the authoritative version of the model.
2. **Perform the forward and backward passes in FP16** (or BF16). Activations, intermediate computations, and gradients are all computed at lower precision. This is faster and uses less memory.
3. **Perform the weight update in FP32.** The low-precision gradients are cast back to FP32 and applied to the FP32 master weights.

**Why this works:** The forward and backward passes process batches of data that are inherently noisy. The direction of the gradient update does not need fine-grained precision -- it just needs to point roughly in the right direction. The weights themselves, however, accumulate many small updates over the course of training. If the weights were stored in low precision, rounding errors would compound over thousands of update steps, degrading the model. Keeping the master weights in FP32 prevents this accumulation.

**Which layers to apply this to** is a design choice. Research has explored applying mixed precision selectively to different layers, and results vary by setup. The general principle is that not all parts of the network are equally sensitive to precision reduction.

---

## 9. Supervised Fine-Tuning (SFT)

### 9.1 The Motivation: From Language Model to Helpful Assistant

A pre-trained model has learned the structure of language and accumulated broad knowledge, but it is not yet useful as an assistant. If you ask a pre-trained base model "Can I put my teddy bear in the washer?", it will not answer your question. Instead, it will continue the text in whatever pattern seems statistically likely -- perhaps describing the materials teddy bears are made of, or generating a paragraph that continues the topic without actually addressing the query. This is because the model was trained on next-token prediction, not on being helpful.

Supervised fine-tuning transforms the pre-trained model into one that responds to instructions and provides helpful answers.

### 9.2 How SFT Works

SFT is a form of continued training where the model is given pairs of (input, desired output) and trained to produce the output given the input.

**Key difference from pre-training:** During pre-training, the loss is computed over every token in the sequence. During SFT, the loss is computed **only over the output tokens**. The input (the user's instruction) is treated as context -- the model processes it but is not penalized for "predicting" it. Teacher forcing is not applied to the input portion. The model learns to generate the appropriate response conditioned on the instruction.

Concretely, given an instruction "Do X" followed by a desired response, the training objective is:

- Process the instruction tokens (no loss computed here).
- Starting from the first response token, compute the next-token prediction loss against the ground-truth response.
- Backpropagate and update weights.

The model starts from its pre-trained weights and further trains them on this supervised data to produce a fine-tuned model.

### 9.3 Instruction Tuning

Instruction tuning is a specific subcategory of SFT aimed at making the model follow instructions and behave as a general-purpose assistant. The data consists of instruction-response pairs across many task categories:

- **Story writing and creative generation**
- **Poem creation**
- **List generation**
- **Explanations and question answering**
- **Mathematical reasoning**
- **Code generation**
- **And many more**

The model learns the general pattern of "user asks, model responds helpfully" rather than being specialized to any single task. This is what enables models like ChatGPT to handle diverse queries.

### 9.4 Data for Instruction Tuning

**Sources of SFT data:**

- **Human-written examples:** Originally, all instruction-tuning data was created by expert linguists and annotators who wrote high-quality responses following detailed guidelines for fluency, helpfulness, and accuracy. This is expensive and time-consuming.
- **LLM-generated examples:** Modern pipelines use existing strong LLMs to generate candidate responses to instructions, then have humans (or other LLMs) review and filter for quality. This speeds up data curation significantly.
- **Safety data:** A critical subset of the training mixture consists of examples designed to make the model refuse harmful requests. This includes data where the model learns to reject dangerous queries and hedge on uncertain claims rather than making blanket statements. Safety behavior is embedded into the model through training, not through regex-based filters (which are not scalable).

**Scale comparison with pre-training:**

| Model | Pre-Training Tokens | SFT Examples |
|---|---|---|
| GPT-3 | 300 billion tokens | ~13,000 examples |
| LLaMA 3 | 15 trillion tokens | ~10 million examples |

Estimating roughly 1,000 tokens per SFT example, the SFT data is several orders of magnitude smaller than the pre-training data. The mental model: pre-training uses massive, raw data to learn language; SFT uses much smaller, high-quality data to align the model's behavior with user expectations.

### 9.5 SFT Challenges

- **Data quality requirements:** SFT data must be high quality. Sloppy or incorrect responses in the training data degrade the model's behavior. Human involvement in the loop remains essential even when using LLM-generated data.
- **Prompt distribution mismatch:** The distribution of instructions in the SFT training set may not match the distribution of prompts users actually submit at inference time. If the training data contains stories about textbook-style poetry but users ask about movie-plot-style stories, the model may struggle to generalize. Aligning the training prompt distribution with the expected inference distribution is critical.
- **Evaluation difficulty:** Quantifying how "good" a fine-tuned model is remains an open problem (discussed in the next section).

---

## 10. Evaluation Benchmarks and Challenges

### 10.1 Standard Benchmarks

Model quality is evaluated across several dimensions, each with its own set of benchmarks:

| Dimension | Benchmark Examples |
|---|---|
| General language understanding | MMLU (Massive Multitask Language Understanding, ~50 tasks) |
| Reasoning | Various reasoning benchmarks |
| Math reasoning | GSM8K (grade-school math, 8,000 examples) |
| Code generation | HumanEval, MBPP, and others |

New benchmarks emerge regularly because models tend to optimize for existing ones, exposing gaps that need new evaluation methods.

### 10.2 The Training-on-Test-Task Problem

A notable phenomenon: models sometimes show sudden performance spikes on specific benchmarks without a clear architectural explanation. Research has revealed that this is often caused by **training on the test task** -- not the exact test set, but data that closely resembles the benchmark's task domain.

If a model's SFT data mixture includes substantial math-reasoning examples and another model's does not, comparing their GSM8K scores is misleading. The difference reflects the data mixture, not the model's intrinsic capability. When comparing models, you must account for whether both were trained on similar task distributions. Benchmarks often include "auxiliary training" datasets that standardize what task-relevant data models are allowed to train on, enabling fairer comparisons.

### 10.3 Chatbot Arena and User Preference

Because benchmarks cannot fully capture user satisfaction, the community developed Chatbot Arena (also known as LMSYS Arena) -- a platform where:

1. Users submit prompts.
2. They receive responses from two anonymous models.
3. They choose which response they prefer.
4. Pairwise comparisons are aggregated into an overall ranking using ELO-like scoring.

This attempts to put a number on the subjective "vibes" -- how helpful and natural the model feels to real users.

**Limitations of Chatbot Arena:**

- **Noisy early rankings:** When a new model enters, the first few matchups disproportionately influence its ranking, introducing instability.
- **Susceptibility to manipulation:** Research has demonstrated that the leaderboard can be rigged. Models can be identified by asking "Who are you?" and an adversarial player could selectively choose which model to favor.
- **User expertise gaps:** Users may prefer responses that are detailed and actionable but factually incorrect over responses that are correct but terse. Expert-curated benchmarks catch factual errors; user preference votes often do not. The teddy bear example illustrates this: a user might prefer a confident but wrong answer about machine-washing over a correct but less detailed answer about hand-washing.
- **Preference bias:** Preferences are subjective and vary across populations. Some users prefer emoji-heavy responses; others strongly dislike them. The demographic distribution of Arena voters may not match the broader user population.
- **Safety penalty in preference:** Users generally dislike when models refuse to answer. This creates a bias toward models that answer everything (even when they should refuse) and against models that enforce appropriate safety boundaries.

### 10.4 The Bottom Line on Evaluation

No single number captures model quality. Effective evaluation requires examining multiple dimensions -- benchmark scores, user preference, safety behavior, factual accuracy -- and matching them to your specific use case. A model that tops the leaderboard on code generation may underperform on creative writing, and vice versa.

---

## 11. Alignment: The Full Picture

The combination of fine-tuning (SFT) and a subsequent step called **preference tuning** (covered in Lecture 5) is collectively called **alignment**. Alignment is everything that happens after pre-training to make the model behave as intended -- following instructions, being helpful, and refusing harmful requests.

A recently emerging step called **mid-training** sits between pre-training and fine-tuning. Mid-training uses the same next-token prediction objective as pre-training but on a curated dataset aligned with the tasks the model will ultimately be used for. It bridges the gap between raw language modeling and task-specific fine-tuning.

The full training pipeline is therefore:

$$\text{Pre-training} \to \text{(Mid-training)} \to \text{SFT} \to \text{Preference Tuning}$$

---

## 12. LoRA (Low-Rank Adaptation)

### 12.1 The Problem

Full fine-tuning updates every parameter in the model. For a model with hundreds of billions of parameters, this is computationally expensive and requires enormous memory (you must store gradients and optimizer states for every parameter).

### 12.2 The Core Idea

LoRA decomposes the weight update during fine-tuning into a low-rank matrix product. Instead of directly modifying a weight matrix $W_0 \in \mathbb{R}^{d \times d}$, LoRA freezes $W_0$ and adds a low-rank correction:

$$W = W_0 + BA$$

where:
- $W_0 \in \mathbb{R}^{d \times d}$ is the frozen pre-trained weight matrix
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ are the trainable LoRA matrices
- $r$ is the rank, typically very small (e.g., 4-16) compared to $d$ (which is typically hundreds to thousands)

**Parameter savings:** The original weight matrix has $d \times d$ parameters. The LoRA matrices have $d \times r + r \times d = 2dr$ parameters. Since $r \ll d$, this is a reduction by a factor of roughly $d / (2r)$. For $d = 4096$ and $r = 4$, LoRA trains only $\sim$0.2% of the parameters that full fine-tuning would require.

### 12.3 How Training Works

During fine-tuning:
1. The forward pass computes $W_0 x + BAx$ for each input $x$. Both terms are computed and summed.
2. Only $B$ and $A$ receive gradient updates. $W_0$ remains frozen.
3. After fine-tuning, the task-specific knowledge is encoded entirely in $B$ and $A$.

### 12.4 Task-Specific Adapters

A powerful property of LoRA is that $B$ and $A$ are task-specific while $W_0$ is shared. You can fine-tune separate LoRA matrices for spam detection, sentiment analysis, code generation, etc., all starting from the same pre-trained $W_0$. At inference time, you swap in the appropriate $(B, A)$ pair for the task at hand. This is far more storage-efficient than maintaining separate full copies of the model for each task.

### 12.5 Where to Apply LoRA

The original LoRA paper applied the low-rank decomposition only to the attention weight matrices ($W_Q$, $W_K$, $W_V$, $W_O$). Subsequent research found that applying LoRA to the **feed-forward blocks** yields the largest performance improvements. Modern practice applies LoRA to both attention and feed-forward layers, but the feed-forward blocks carry the bulk of the benefit.

### 12.6 Training Dynamics

Two empirically observed properties of LoRA training:

1. **Higher learning rates are needed.** LoRA typically uses learning rates roughly 10x larger than full fine-tuning. A plausible explanation is that the low-rank matrices operate in a compressed parameter space and need larger steps to explore it effectively.
2. **Smaller batch sizes work better.** Unlike full fine-tuning (where larger batches generally help), LoRA performs worse with large batch sizes. One hypothesis is that the training dynamics of a product of two matrices ($B \times A$) differ fundamentally from those of a single full matrix, and smaller batches provide better gradient signal in this regime.

### 12.7 The Rank Hyperparameter

The rank $r$ is a design choice. Common values are in the range of 4-16. While you could perform a grid search over $r$, the initial reduction from full fine-tuning to LoRA is so dramatic (orders of magnitude fewer parameters) that the marginal impact of choosing $r = 4$ versus $r = 8$ is usually small. In practice, picking a commonly used value and treating it as a standard hyperparameter is sufficient.

---

## 13. QLoRA (Quantized LoRA)

QLoRA combines the quantization techniques from Section 8 with LoRA to further reduce the memory footprint of fine-tuning.

### 13.1 The Approach

In standard LoRA, the frozen weights $W_0$ are stored in their original precision (typically FP16 or BF16). QLoRA quantizes $W_0$ to a much lower precision while keeping the trainable LoRA matrices $A$ and $B$ in full precision (BF16):

$$W = \text{quantize}(W_0) + BA$$

- $W_0$ is stored in **NF4 (4-bit NormalFloat)** format.
- $A$ and $B$ are stored and trained in **BF16**.

### 13.2 NF4 Quantization

NF4 is a quantization format specifically designed for neural network weights, which are typically normally distributed. Rather than dividing the number range into equally spaced bins (as in standard integer quantization), NF4 divides it into **quantiles** of the normal distribution. This ensures that each quantized level contains approximately the same number of weight values, making optimal use of the available 4 bits.

### 13.3 Double Quantization

QLoRA introduces a second quantization step: it quantizes the **quantization constants** themselves (the scaling factors needed to convert between quantized and full-precision formats). This provides a small additional memory saving on top of the primary NF4 quantization.

### 13.4 Memory Savings

The NF4 quantization of frozen weights yields approximately **16x VRAM savings** compared to storing them in FP16 (since you go from 16 bits to 4 bits per weight, a 4x reduction, combined with other optimizations in the paper). The double quantization provides additional but smaller savings. Together, these techniques make it possible to fine-tune very large models on consumer-grade GPUs that would otherwise lack sufficient memory.

---

## Key Takeaways

1. **Transfer learning** is the paradigm behind modern LLMs: pre-train once on vast data to learn language, then fine-tune for specific tasks. This avoids training from scratch for every new application.

2. **Pre-training** is the most expensive stage, consuming trillions of tokens and costing millions of dollars. The model learns next-token prediction on raw text from the internet, code repositories, and other sources. Knowledge is frozen at the cutoff date.

3. **Scaling laws** show that performance improves predictably with more compute, data, and parameters. The **Chinchilla result** establishes that compute-optimal training uses roughly 20 tokens per parameter, revealing that many early models were undertrained.

4. **GPU parallelism** is essential because no single GPU can hold a large LLM and its training state. **Data parallelism** distributes batches across GPUs. **ZeRO** eliminates redundant storage of parameters, gradients, and optimizer states. **Model parallelism** splits the model itself (tensor, pipeline, and expert parallelism).

5. **Flash attention** exploits the GPU memory hierarchy (fast SRAM vs. slow HBM) by processing attention in small tiles that stay in SRAM. A decomposition of softmax across blocks makes this exact. Combined with **recomputation** of activations during the backward pass, it achieves both faster runtime and lower memory usage -- an unusual win on both dimensions.

6. **Mixed precision training** stores master weights in FP32 but performs forward/backward passes in FP16 or BF16. This saves memory and increases throughput with minimal impact on model quality, because gradient directions do not need full precision while accumulated weights do.

7. **Supervised fine-tuning (SFT)** transforms a pre-trained language model into a helpful assistant by training on instruction-response pairs. The loss is computed only on the response, not the instruction. The data is orders of magnitude smaller than pre-training data but must be high quality.

8. **Evaluation** remains an open challenge. Benchmarks (MMLU, GSM8K, etc.) measure specific capabilities but can be gamed through training on similar tasks. User preference platforms like Chatbot Arena capture subjective quality but are noisy, manipulable, and biased toward responses that avoid refusals.

9. **LoRA** makes fine-tuning efficient by freezing pre-trained weights and training only small low-rank matrices ($B$ and $A$) that capture task-specific adaptations. This reduces trainable parameters by orders of magnitude and enables task-specific adapters that share a single base model.

10. **QLoRA** pushes efficiency further by quantizing frozen weights to 4-bit NF4 format while training LoRA matrices in BF16, achieving roughly 16x VRAM savings and enabling fine-tuning of large models on limited hardware.
