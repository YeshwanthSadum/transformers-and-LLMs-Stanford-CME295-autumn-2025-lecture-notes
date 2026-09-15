# Lecture 2: Transformer-Based Models and Tricks - Stanford CME 295

**Date:** October 3, 2025
**Instructors:** Afshine Amidi & Shervine Amidi (Adjunct Lecturers, Stanford)
**Course:** CME 295 - Transformers and Large Language Models, Autumn 2025

---

## 1. Recap of Self-Attention and Multi-Head Attention

The lecture opens with a recap of the core ideas from Lecture 1. Self-attention lets each token attend to every other token in the sequence through the query-key-value mechanism. The self-attention formula is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q$ (queries), $K$ (keys), and $V$ (values) are linear projections of the input embeddings. The query asks "which other tokens are most similar to me?" by computing dot products with all keys, and the resulting weights determine how much of each token's value to incorporate.

### 1.1 Multi-Head Attention Revisited

In the original Transformer, the attention layer uses multiple heads. Each head has its own projection matrices $W_Q^{(i)}$, $W_K^{(i)}$, and $W_V^{(i)}$, giving the model multiple independent "opportunities" to learn different ways of projecting inputs into queries, keys, and values. The outputs of all heads are concatenated and projected through an output matrix $W_O$.

### 1.2 Interpreting Attention Maps

The "Attention Is All You Need" paper includes visualizations called **attention maps** that help interpret what each head learns. An attention map shows, for a given token, the attention weights it assigns to every other token. For example, when examining the token "its" in a sentence, the attention map highlights "law" and "application" as receiving high attention weights -- which makes sense because "its" refers back to those words. Different heads highlight different tokens, suggesting that heads specialize in capturing different types of linguistic relationships (e.g., one head might capture coreference, another syntactic dependency). These attention weights are the dot products $q_i \cdot k_j$ after softmax normalization.

The computation across heads is fully parallelized. Each head independently computes its own query-key-value projections, performs the attention computation, and produces a result. These results are then concatenated and projected through $W_O$ to produce the final output.

---

## 2. Position Embeddings

The Transformer architecture processes all tokens simultaneously through self-attention, which means it has no inherent notion of word order. Unlike RNNs, which process tokens sequentially and therefore implicitly encode position, the Transformer needs an explicit mechanism to inject positional information. Without it, "dog bites man" and "man bites dog" would produce identical representations.

### 2.1 Learned Position Embeddings

The simplest approach: assign a learnable embedding vector to each position (position 1, position 2, ..., up to some maximum). These embeddings are added element-wise to the corresponding token embeddings. For "A cute teddy bear is reading", the embedding for "A" becomes the token embedding of "A" plus the learned embedding for position 1.

**How training works:** You initialize placeholder embedding vectors for each position (e.g., positions 1 through 512) and let standard gradient descent learn their values during training, just like any other model parameter.

**Limitations:**

- **Overfitting to training data:** The learned embeddings reflect whatever positional patterns exist in the training set. If the training data consistently has certain structures at certain positions, the embeddings will encode those biases.
- **Cannot extrapolate beyond training length:** If the model is trained on sequences of length 512, it has no learned embedding for position 513. At inference time, encountering longer sequences requires some form of extrapolation, which this method does not naturally support.

**Advantage:** Gradient descent is remarkably effective at finding useful representations from data, so these embeddings tend to work well within the lengths seen during training.

### 2.2 Sinusoidal Position Embeddings

The original Transformer paper proposed an alternative: instead of learning position embeddings, compute them using a fixed formula based on sine and cosine functions. For a given position $m$, the embedding is a vector of dimension $d_{\text{model}}$, where each component is:

$$PE_{(m, 2i)} = \sin(\omega_i \cdot m), \quad PE_{(m, 2i+1)} = \cos(\omega_i \cdot m)$$

where:

$$\omega_i = 10000^{-2i/d_{\text{model}}}$$

- $m$ is the position index in the sequence
- $i$ is the dimension index in the embedding vector
- $d_{\text{model}}$ is the total embedding dimension (must match the token embedding dimension since they are added together)
- $\omega_i$ controls the frequency of oscillation for dimension $i$

**Why sine and cosine make sense.** The core intuition is that we want tokens closer together to have more similar position embeddings than tokens far apart. Consider the dot product of two position embeddings at positions $m$ and $n$. Using the trigonometric identity:

$$\cos(\omega_i(m - n)) = \cos(\omega_i m)\cos(\omega_i n) + \sin(\omega_i m)\sin(\omega_i n)$$

The right side is exactly one term of the dot product between the position embeddings at $m$ and $n$ (each pair of sine-cosine components contributes one such term). The full dot product is therefore a sum of cosines:

$$PE_m \cdot PE_n = \sum_i \cos(\omega_i(m - n))$$

This dot product depends only on the **relative distance** $m - n$, not on the absolute positions. Since $\cos(0) = 1$ is the maximum of cosine, the dot product is maximized when $m = n$ (a position is most similar to itself), and it decreases as $|m - n|$ grows -- matching the intuition that nearby tokens should have more similar positional representations.

We care about the dot product because similarity in embedding space is measured via the dot product (cosine similarity is just a normalized dot product).

**Frequency structure:** When you visualize the sinusoidal embeddings as a matrix (positions on one axis, dimensions on the other), a pattern emerges. Low dimensions have high-frequency oscillations (values change rapidly across positions), while high dimensions have low-frequency oscillations (values change slowly). This is because $\omega_i$ is large for small $i$ and small for large $i$. The result is a rich encoding where different dimensions capture positional patterns at different scales.

**Key advantage over learned embeddings:** Sinusoidal embeddings can extend to any sequence length at inference time, since the formula can be evaluated for any value of $m$. The original paper found that sinusoidal embeddings perform comparably to learned embeddings, with the added benefit of length generalization.

### 2.3 Injecting Position Information Directly into Attention

Both learned and sinusoidal embeddings share a design choice: they are added to the input embeddings before the attention layers. But what we actually care about is how positions affect the **attention computation** -- specifically the similarity scores $QK^\top$ inside the softmax. Adding positional information at the input is an indirect way to influence attention; the positional signal must survive multiple transformations before reaching the attention layer.

This motivated a shift toward methods that inject positional information **directly into the attention formula**:

$$\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + B\right) V$$

where $B$ is a bias matrix whose entries reflect the relative positions of token pairs. Several methods implement this idea differently.

### 2.4 T5 Relative Position Bias

The T5 model (Text-to-Text Transfer Transformer) learns a **relative position bias** term that is added inside the softmax. The relative distance $m - n$ between two positions is mapped into a set of discrete buckets, and each bucket has a learned bias value. These learned biases are then injected into the attention computation.

This approach directly modulates attention scores based on relative position. Adding a bias inside the softmax is valid because softmax normalizes its inputs -- making the final attention weights sum to 1 regardless of what biases are added. A more negative bias for distant token pairs will cause them to receive lower attention weights after normalization.

**Limitation:** As with any learned approach, the biases depend on the training data.

### 2.5 ALiBi (Attention with Linear Biases)

ALiBi, introduced in the "Train Short, Test Long" paper, replaces learned biases with a deterministic formula. The bias for positions $m$ and $n$ is simply a function of their relative distance $|m - n|$, scaled by a head-specific constant. This avoids the learnable component entirely, using a straightforward linear penalty that increases with distance.

While ALiBi showed good results and supports length generalization, most modern models have converged on a different approach.

### 2.6 RoPE (Rotary Position Embeddings)

RoPE is the position embedding method used by most modern language models. Instead of adding a bias to the attention scores, it **rotates** the query and key vectors by angles that depend on their positions.

**Core idea in 2D:** Consider a query vector $\mathbf{q}$ at position $m$ and a key vector $\mathbf{k}$ at position $n$, both in 2D space. RoPE rotates $\mathbf{q}$ by angle $\theta \cdot m$ and $\mathbf{k}$ by angle $\theta \cdot n$ using the standard 2D rotation matrix:

$$R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}$$

Multiplying a vector $\mathbf{v} = (x, y)^\top$ by $R(\alpha)$ rotates it by angle $\alpha$ in the 2D plane. This can be verified by expressing $\mathbf{v}$ in polar coordinates as $r(\cos\phi, \sin\phi)^\top$ and showing that $R(\alpha)\mathbf{v} = r(\cos(\phi + \alpha), \sin(\phi + \alpha))^\top$ -- the vector's magnitude is preserved and its angle increases by $\alpha$.

**Why rotation gives relative position information:** When computing the attention score between the rotated query and key:

$$\text{score} = (R(\theta m)\,\mathbf{q})^\top (R(\theta n)\,\mathbf{k})$$

The resulting expression involves the rotation matrix $R(\theta(n - m))$. The attention score becomes a function of the **relative distance** $n - m$, not the absolute positions $m$ and $n$ individually. This is the same desirable property we saw with sinusoidal embeddings, but now achieved directly inside the attention computation rather than at the input.

**Extension to higher dimensions:** In practice, embeddings have dimension $d_{\text{model}} \gg 2$. RoPE handles this by applying independent 2D rotations to consecutive pairs of dimensions. The full rotation matrix is block-diagonal, with each $2 \times 2$ block being a rotation matrix at a different frequency $\theta_i$. The frequencies $\theta_i$ follow a similar pattern to the sinusoidal embeddings:

$$\theta_i \approx 10000^{-2i/d_{\text{model}}}$$

where $i$ ranges from 1 to $d_{\text{model}}/2$.

**Long-term decay property:** The upper bound of the attention weight between two tokens decreases as the distance $|m - n|$ grows. While there are small oscillations, the mathematical analysis in the RoPE paper shows a clear long-term decay trend, confirming that distant tokens receive lower attention weights on average.

**Why RoPE dominates modern practice:** It combines the best properties of prior methods -- it depends only on relative distance (not absolute position), operates directly in the attention computation (not at the input), uses a deterministic formula (no learned parameters to overfit), and draws on the sinusoidal intuition from the original Transformer paper.

---

## 3. Layer Normalization

The Transformer architecture includes "Add & Norm" blocks after each sub-layer (self-attention and feed-forward network). These perform a residual connection (add) followed by normalization (norm). Normalization is a technique that helps with training stability and speeds up convergence.

### 3.1 Layer Norm in the Original Transformer

Given a vector $\mathbf{x}$ (the activation at some point in the network), layer normalization computes:

$$\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sigma} + \beta$$

where:

- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ is the mean of the vector's components
- $\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2}$ is the standard deviation
- $\gamma$ and $\beta$ are learned parameters (scale and shift)

**Why normalize?** As data flows through the layers of a deep network, the distribution of activations can shift dramatically -- some components might become very large while others are very small. This phenomenon, known as **internal covariate shift**, makes it difficult for subsequent layers to learn stable weights. Layer normalization brings each activation vector's components into a standardized range, making optimization smoother and faster.

**Layer norm vs. batch norm:** Batch normalization normalizes each dimension across the batch (i.e., for a given dimension, compute mean and variance over all examples in the batch). Layer normalization normalizes across the dimensions within a single example. For Transformer-based models, layer normalization is preferred because: (1) it is empirically more effective; and (2) batch normalization introduces a dependency on the batch composition, which can cause discrepancies between training (where you have a batch) and inference (where you may have a single example).

### 3.2 Pre-Norm vs. Post-Norm

The original Transformer uses **post-norm**: the sub-layer output and input are summed first, then normalized. Schematically:

$$\mathbf{y} = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$$

Modern architectures use **pre-norm**: the input is normalized before entering the sub-layer, and the residual is added after:

$$\mathbf{y} = \mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))$$

Pre-norm has been found to improve training stability, especially for deep models.

### 3.3 RMSNorm (Root Mean Square Normalization)

Modern models have further simplified layer normalization by replacing it with RMSNorm:

$$\text{RMSNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

where $\gamma$ is a learned scaling parameter. Compared to standard layer normalization, RMSNorm drops the mean subtraction and the $\beta$ shift term, learning only $\gamma$. This makes it computationally cheaper with fewer parameters, while achieving comparable training dynamics.

---

## 4. Sparse Attention and Sliding Window Attention

Full self-attention has $O(n^2)$ complexity in the sequence length $n$, because every token computes attention scores with every other token. As sequences grow longer, this quadratic cost becomes prohibitive.

### 4.1 Local Attention

The Longformer paper (2020) explored restricting the attention window. Instead of letting each token attend to the entire sequence, each token only attends to a local neighborhood. This converts the attention pattern from a full $n \times n$ matrix to a banded structure.

In practice, this is implemented efficiently through tiling and other computational tricks that avoid materializing the full attention matrix. The key insight is that you never compute the full $\text{softmax}(QK^\top / \sqrt{d_k})$ -- instead, you only compute the attention scores within each local window.

### 4.2 Sliding Window Attention

The modern term for local attention is **sliding window attention**. Each token attends only to tokens within a fixed-size window around it. While the illustrative diagrams show small windows, in practice the window size can be several thousand tokens -- substantial enough to capture most local context.

**Connection to convolutional receptive fields:** With stacked layers of sliding window attention, information can propagate beyond a single window. A token at layer $l$ attends to its neighbors, but those neighbors already incorporated information from their neighbors at layer $l-1$. After multiple layers, a token's **effective receptive field** spans much of the sequence, analogous to how stacking convolutional layers in computer vision expands the receptive field. The Mistral model, for example, uses sliding window attention at every layer and relies on this stacking effect for long-range information flow.

### 4.3 Interleaving Local and Global Attention

Modern architectures do not use exclusively local or exclusively global attention. Instead, they interleave layers: some layers use sliding window (local) attention, while others use full (global) attention. The specific pattern varies by model -- there is no single recipe, but the combination gives both computational efficiency and the ability to capture long-range dependencies.

---

## 5. Sharing Attention Heads: MHA, MQA, and GQA

Another axis of variation in modern Transformers is how projection matrices are shared across attention heads. This is motivated by both computational cost and memory efficiency, particularly during autoregressive decoding.

### 5.1 The KV Cache Problem

During autoregressive generation (decoding), each new token must attend to all previously generated tokens. The keys and values for all prior tokens are recomputed at every step unless they are cached. This **KV cache** stores the key and value projections for all prior tokens, avoiding redundant computation. However, the cache grows with sequence length and with the number of heads, consuming significant memory.

Sharing key and value projection matrices across heads reduces the size of the KV cache proportionally, which is one of the primary motivations for the methods below.

### 5.2 Multi-Head Attention (MHA)

The standard approach from the original Transformer: each of the $h$ heads has its own independent query, key, and value projection matrices ($W_Q^{(i)}$, $W_K^{(i)}$, $W_V^{(i)}$ for $i = 1, \dots, h$). This provides maximum expressiveness -- each head can learn entirely different projections -- but also maximum memory usage for the KV cache.

### 5.3 Multi-Query Attention (MQA)

The extreme sharing strategy: all $h$ heads share a **single** key projection matrix and a **single** value projection matrix. Each head retains its own query projection matrix. This drastically reduces the KV cache size (by a factor of $h$) and computation, at some cost to model expressiveness.

**Why keep separate queries but share keys and values?** The query represents "what am I looking for?" and maintaining diverse queries lets each head ask different questions. The keys and values, on the other hand, represent "what do I contain?" and "what information do I provide?" -- sharing these is less harmful because the diversity in what each head attends to is already captured by the diverse queries.

### 5.4 Grouped-Query Attention (GQA)

A middle ground between MHA and MQA. The $h$ heads are divided into $g$ groups. Each group of $h/g$ heads shares one set of key and value projection matrices, while every head still has its own query projection. When $g = 1$, this reduces to MQA. When $g = h$, it becomes standard MHA.

GQA is the most common choice in recent models because it balances expressiveness with memory efficiency. The choice of $g$ depends on the model size, target latency, and how much KV cache memory is acceptable.

### 5.5 Choosing Between Them

The choice is driven by performance requirements, latency constraints, memory budget, and input sequence length. Longer sequences create larger KV caches, making sharing more attractive. Smaller models may use MQA for efficiency; larger models can afford GQA with more groups. There is no universal answer, but the trend in recent models is toward GQA.

---

## 6. Transformer-Based Model Families

The original Transformer (2017) used an encoder-decoder architecture for machine translation. Since then, three families of models have emerged, each using a different subset of the architecture.

### 6.1 Encoder-Decoder Models

These retain both the encoder and decoder from the original Transformer. The encoder processes the input sequence, and the decoder generates the output sequence using cross-attention to the encoder's representations.

**T5 (Text-to-Text Transfer Transformer):** The most prominent encoder-decoder family. T5 frames every NLP task as a text-to-text problem. It introduced several variants:

- **T5** (vanilla): The base model.
- **mT5** (multilingual T5): Trained on multilingual data with a vocabulary covering many languages.
- **ByT5** (byte-level T5): Operates at the byte level instead of using a learned tokenizer. The vocabulary size drops from roughly 30,000 to $2^8 = 256$ (one entry per byte, with each character representable in two bytes). This eliminates tokenization entirely, at the cost of much longer sequences.

**Span corruption objective:** T5 uses a different pre-training objective than standard next-token prediction. Given an input sentence, one or more spans of tokens are replaced with **sentinel tokens** (special placeholder tokens). For example, "my teddy bear is cute and reading" might become "my [sentinel_1] is [sentinel_2] and reading", where [sentinel_1] replaces "teddy bear" and [sentinel_2] replaces "cute". The encoder receives this corrupted input, and the decoder must reconstruct the missing spans in sequence -- outputting [sentinel_1] followed by "teddy bear", then [sentinel_2] followed by "cute", and so on.

**Training uses teacher forcing:** During training, the decoder receives the full expected output and learns to reconstruct all spans simultaneously, rather than generating one token at a time.

### 6.2 Encoder-Only Models

These drop the decoder entirely and use only the encoder. Without a decoder, these models cannot perform autoregressive text generation. Instead, they produce rich, context-aware representations of the input that are well-suited for **classification tasks** -- sentiment analysis, named entity recognition, question answering (extractive), and similar.

The key examples are **BERT**, **DistilBERT**, and **RoBERTa**, all covered in detail in Section 7.

### 6.3 Decoder-Only Models

These drop the encoder entirely and use only the decoder (with masked self-attention and a feed-forward network, but no cross-attention since there is no encoder to attend to). This is the architecture behind modern LLMs like GPT, LLaMA, and their successors.

**Why decoder-only became dominant:**

- **Simplicity of objective:** Next-token prediction is the simplest possible training task -- predict the next word given all preceding words. It requires no special data preparation (no span corruption, no sentence pairing). This simplicity makes it easy to scale to massive datasets.
- **Alignment with chatbot applications:** The autoregressive generation paradigm naturally fits conversational use cases, which are the primary application of today's LLMs.
- **Compute efficiency:** Research showed that investing the entire compute budget into a decoder-only model outperforms splitting it between encoder and decoder components.

The trajectory of the field moved from encoder-decoder dominance (2017-2019) to encoder-only for classification (BERT era, 2018-2020) to decoder-only for general-purpose language modeling (2020-present). Decoder-only models will be the central topic of the next lectures.

---

## 7. BERT: Bidirectional Encoder Representations from Transformers

BERT is a landmark encoder-only model published in 2018 with roughly 170,000 citations. It demonstrated that pre-training on unlabeled text followed by fine-tuning on a small labeled dataset could achieve state-of-the-art results across a wide range of NLP classification tasks.

### 7.1 Bidirectionality

The "bidirectional" in BERT's name refers to the fact that, because it uses only the encoder (with standard self-attention, not masked self-attention), every token attends to every other token in the sequence -- both before and after it. This is in contrast to decoder models (like GPT), where the causal mask restricts each token to attending only to preceding tokens.

This distinction matters: a bidirectional model builds representations that incorporate context from the entire input, while a unidirectional (causal) model can only incorporate context from the left. For classification tasks, bidirectional context is strictly more informative.

### 7.2 Historical Context: ELMo

BERT was published the same year as **ELMo (Embeddings from Language Models)**, another model that produced bidirectional, context-dependent word representations. ELMo used a bidirectional LSTM with multiple stacked layers. However, it was based on recurrent architecture and therefore suffered from the same scalability problems as all RNNs -- the sequential processing bottleneck made it difficult to scale to large datasets and long sequences. BERT, built on the Transformer, could leverage parallelism and scaled far more effectively. ELMo was overshadowed as a result, despite introducing genuinely novel ideas about bidirectional representations.

### 7.3 Architecture

BERT takes the encoder portion of the original Transformer and adds structural tokens and new pre-training objectives.

**Special tokens:**

- **[CLS] (classification):** A special token prepended to every input sequence. After passing through all encoder layers, the output embedding of [CLS] serves as a single vector representing the entire input sequence. This is the embedding that gets fed into classification heads during fine-tuning.
- **[SEP] (separator):** Placed between two sentences when the input consists of a sentence pair (used for the NSP task). Also placed at the end of the input.
- **[PAD] (padding):** Used to pad sequences to a fixed length within a batch, since batched training requires all sequences to have the same length.

**Tokenizer:** BERT uses **WordPiece**, a subword tokenization method that learns merge rules maximizing the likelihood of the training data. The vocabulary size is approximately 30,000 tokens.

### 7.4 Input Representation

Each token's input to the encoder is the element-wise sum of three embeddings:

1. **Token embedding:** A learned embedding from a lookup table, mapping each vocabulary token to a dense vector. This is the same as in the original Transformer.
2. **Position embedding:** Encodes the token's position in the sequence. The original BERT paper uses a fixed (sinusoidal) version, though learned versions perform comparably.
3. **Segment embedding:** A new addition specific to BERT. There are only two segment embeddings (segment A and segment B). All tokens in the first sentence receive segment A's embedding; all tokens in the second sentence receive segment B's embedding. This helps the model distinguish which sentence each token belongs to, which is relevant for the NSP pre-training task. Segment embeddings are learned through gradient descent.

### 7.5 Pre-Training: Two-Stage Training

BERT uses a **two-stage** training process. The first stage (pre-training) learns general-purpose representations from unlabeled data. The second stage (fine-tuning) adapts those representations to a specific downstream task using a small amount of labeled data.

**Pre-training uses two objective functions simultaneously:**

#### Masked Language Model (MLM)

A random subset of input tokens (15%) is selected for prediction. Of these selected tokens:

- **80%** are replaced with a special [MASK] token
- **10%** are left unchanged (the model must still predict the original token)
- **10%** are replaced with a random word from the vocabulary

The model must predict the original identity of each selected token based on the context provided by all surrounding tokens (both left and right, since attention is bidirectional). This forces the model to learn deep contextual representations -- to predict a masked word, the model must understand what words make sense in that context.

The 80/10/10 split prevents the model from learning a shortcut. If masking were always applied, the model might learn to only pay attention to [MASK] tokens and ignore everything else. By sometimes leaving tokens unchanged or replacing them randomly, the model must maintain robust representations for all tokens.

#### Next Sentence Prediction (NSP)

Two sentences are presented as input (separated by [SEP]). 50% of the time, the second sentence genuinely follows the first in the training corpus. The other 50%, the second sentence is randomly sampled from elsewhere in the corpus. A binary classification head on top of the [CLS] token predicts whether the two sentences are consecutive.

The hypothesis was that this task helps the model learn relationships between sentences, which would be useful for downstream tasks like question answering and natural language inference.

**Pro: Self-supervised.** Both MLM and NSP require no human annotation. For MLM, the labels are the original tokens (which you know because you masked them). For NSP, the labels are determined by whether you drew the second sentence from the correct position or randomly. This means pre-training can leverage enormous amounts of unlabeled text.

### 7.6 Model Configurations

The BERT paper uses slightly different notation from the original Transformer: $L$ for the number of layers (was $N$), $H$ for the hidden dimension (was $d_{\text{model}}$), and $A$ for the number of attention heads (was $h$).

| Configuration | Layers ($L$) | Hidden dim ($H$) | Attention heads ($A$) | Parameters |
|---|---|---|---|---|
| BERT-Base | 12 | 768 | 12 | ~110M |
| BERT-Large | 24 | 1024 | 16 | ~340M |

**Cased vs. uncased:** BERT is available in "cased" (preserves original casing) and "uncased" (lowercases all text during preprocessing) variants. The choice depends on whether case information matters for the downstream task (e.g., NER benefits from casing since proper nouns are capitalized).

### 7.7 Fine-Tuning

After pre-training, the encoder weights are frozen (or partially frozen) and a small task-specific network is attached on top. The model is then trained on labeled data for the target task.

**Sentence-level tasks (e.g., sentiment analysis):** A classification layer (FFN) is placed on top of the [CLS] token's output embedding. The FFN typically consists of two linear projections with a hidden layer, mapping from the embedding dimension to the number of target classes. Only the FFN weights are trained (or the entire model is fine-tuned end-to-end with a low learning rate).

**Token-level tasks (e.g., question answering, NER):** Classification heads are placed on top of each token's output embedding. For extractive question answering, two separate FFNs predict the start and end positions of the answer span within the input text. In this case, the output embeddings of all tokens are used -- not just [CLS].

The fine-tuning dataset can be quite small because the pre-trained encoder already produces high-quality, contextual embeddings. The fine-tuning stage only needs to learn the final projection from these embeddings to the task-specific labels.

**The [CLS] token works for sentence-level classification** because the self-attention mechanism has mixed information from all other tokens into its representation across all encoder layers. By the final layer, the [CLS] embedding is a context-aware summary of the entire input, making it suitable as a single-vector representation for classification.

### 7.8 Strengths and Limitations

**Strengths:**

- **Contextual embeddings:** Each token's representation depends on the full surrounding context, enabling the model to disambiguate words based on usage (unlike Word2Vec, where each word has a single fixed embedding).
- **Transfer learning:** The pre-train then fine-tune paradigm means the expensive pre-training step is done once, and the model can be cheaply adapted to many different downstream tasks.
- **Data efficiency:** Fine-tuning requires very little labeled data because the pre-trained representations are already highly informative.
- **Widely used in industry:** BERT and its variants remain the standard for classification-oriented NLP tasks in production systems (sentiment detection, content moderation, entity extraction, etc.).

**Limitations:**

- **No text generation:** Without a decoder, BERT cannot generate text autoregressively.
- **Limited context length:** The original BERT supports a maximum sequence length of 512 tokens. Longer inputs must be truncated or processed in chunks. (Later variants like Longformer address this using sparse attention techniques.)
- **Two-stage training overhead:** The pre-training plus fine-tuning pipeline is more complex than single-stage training approaches.

---

## 8. Extensions of BERT

### 8.1 DistilBERT: Knowledge Distillation for Efficiency

BERT-Base has 110 million parameters, which can be too costly for latency-sensitive production deployments. **DistilBERT** addresses this by applying **knowledge distillation** to compress the model.

**Knowledge distillation** is based on the insight (from Hinton, Vinyals, and Dean) that the output probability distribution of a trained model contains far more information than hard labels alone. When a teacher model assigns 70% probability to the correct class and 20% to a related class, that 20% encodes meaningful information about the structure of the problem -- something a hard label (just "correct" or "incorrect") discards.

The distillation process trains a smaller **student** model to match the output distribution of a larger **teacher** model, rather than matching hard labels. The objective minimizes the **KL divergence** between the teacher's and student's output distributions:

$$\mathcal{L}_{\text{KL}} = \sum_i P_T(i) \log \frac{P_T(i)}{P_S(i)}$$

where $P_T$ is the teacher's output distribution and $P_S$ is the student's. This loss measures how much information is lost when using the student's distribution to approximate the teacher's. When the teacher's distribution degenerates to a hard label (one-hot), the KL divergence reduces to the standard cross-entropy loss $-\log P_S(\text{correct class})$.

**DistilBERT's key finding:** Reducing the number of encoder layers by half (from 12 to 6) and using distillation to retain performance yields a model that is 60% smaller and 60% faster, while retaining approximately 97% of BERT's performance. The paper is notably concise (4 pages) and highly impactful.

### 8.2 RoBERTa: Robustly Optimized BERT

RoBERTa revisited BERT's design decisions and found several improvements:

- **Dropping NSP:** Removing the next sentence prediction objective led to no decrease (and sometimes an improvement) in downstream performance. This challenged the original BERT authors' hypothesis that NSP helps learn useful inter-sentence representations.
- **Dynamic masking:** Instead of generating the masked version of each training example once during data preprocessing (static masking), RoBERTa applies different random masks each time a training example is seen across epochs. This increases the diversity of training signal.
- **More data and longer training:** The original BERT was substantially undertrained. RoBERTa increased both the diversity and volume of training data and trained for more steps, leading to significant benchmark improvements.

These changes are straightforward but collectively produced meaningful gains, demonstrating that BERT's original training recipe left substantial performance on the table.

---

## Key Takeaways

1. **Position embeddings** solve the Transformer's lack of inherent word order. The field progressed from learned embeddings (limited to training-length sequences) to sinusoidal embeddings (generalizable but injected at the input) to **RoPE** (operates directly in the attention computation via rotation, depends only on relative distance, and is the current standard).

2. **Layer normalization** stabilizes training by controlling activation magnitudes. Modern models use **pre-norm** placement (normalize before the sub-layer, not after) and **RMSNorm** (simpler, fewer parameters, comparable performance) instead of the original post-norm layer normalization.

3. **Sliding window attention** reduces the $O(n^2)$ cost of full self-attention by restricting each token to a local neighborhood. Modern architectures interleave local and global attention layers, and stacking local attention layers creates an expanding receptive field analogous to convolutions in vision.

4. **Grouped-Query Attention (GQA)** shares key and value projections across groups of heads, reducing the KV cache size during decoding. It occupies the practical middle ground between full Multi-Head Attention (MHA) and the extreme sharing of Multi-Query Attention (MQA).

5. Three model families emerged from the Transformer: **encoder-decoder** (T5), **encoder-only** (BERT), and **decoder-only** (GPT and successors). The field has converged on decoder-only architectures for general-purpose LLMs, while encoder-only models remain the standard for classification tasks.

6. **BERT** demonstrated the power of bidirectional pre-training with MLM and NSP, followed by lightweight fine-tuning. Its contextual embeddings, produced via full (unmasked) self-attention, are rich enough that even a simple linear classifier on top can achieve strong performance across diverse NLP tasks.

7. **DistilBERT** showed that knowledge distillation can compress BERT to nearly half its size with minimal performance loss. **RoBERTa** showed that BERT's original training recipe was suboptimal -- dropping NSP, using dynamic masking, and training on more data all improve results.
