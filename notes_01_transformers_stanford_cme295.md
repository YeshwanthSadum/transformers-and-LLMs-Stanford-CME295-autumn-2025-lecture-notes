# Lecture 1: Transformers - Stanford CME 295

**Date:** September 26, 2025
**Instructors:** Afshine Amidi & Shervine Amidi (Adjunct Lecturers, Stanford)
**Course:** CME 295 - Transformers and Large Language Models, Autumn 2025

---

## 1. NLP Task Categories

NLP (Natural Language Processing) is the field concerned with computationally manipulating, understanding, and generating human language. All NLP tasks can be grouped into three broad categories based on the shape of their input and output.

### 1.1 Classification (text in, single label out)

The simplest form: given an input text, predict a single discrete label.

**Common tasks:**
- **Sentiment analysis:** Given a product or movie review, predict whether the sentiment is positive, negative, or neutral. Standard datasets include IMDb movie reviews, Amazon product reviews, and X (Twitter) posts.
- **Intent detection:** Given a user utterance like "I want to create an alarm for tomorrow", identify the user's intent (here: "create alarm"). This is foundational to virtual assistants and chatbots.
- **Language detection:** Given a piece of text, identify which language it is written in (e.g., French, German, English).
- **Topic modeling:** Classify text into broad subject areas.

**Evaluation metrics:**

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

Accuracy alone can be misleading with imbalanced datasets. If 99% of reviews are positive, a model that always predicts "positive" gets 99% accuracy while being useless. This is why we also track:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Precision** answers: "Of everything the model labeled positive, how many were actually positive?" High precision means few false alarms.
- **Recall** answers: "Of everything that actually is positive, how many did the model catch?" High recall means few misses.
- **F1** is the harmonic mean of precision and recall, giving a single number that balances both. It is especially useful when you cannot afford to favor one over the other.

### 1.2 Multi-classification (text in, multiple labels out)

Here the model must assign a label to each token (or span) in the input, not just one label for the whole text.

**Common tasks:**
- **Named Entity Recognition (NER):** Given a sentence, label specific words or spans with their entity type. For example, in "Teddy Bear flew to Paris on Monday", the model should tag "Paris" as LOCATION and "Monday" as TIME. NER is widely used in information extraction pipelines and search engines.
- **Part-of-speech (POS) tagging:** Label each word with its grammatical role (noun, verb, adjective, etc.). This was heavily studied in computational linguistics a decade ago.
- **Dependency / constituency parsing:** Analyze the grammatical structure of sentences by building parse trees.

**Metrics** are computed at the token level or aggregated per entity type (e.g., "how well do we detect LOCATION entities specifically?").

### 1.3 Generation (text in, variable-length text out)

The most flexible and currently the most popular category. The model receives text as input and produces text of arbitrary length as output.

**Common tasks:**
- **Machine translation:** Convert text from one language to another. A popular benchmark dataset is WMT (Workshop on Machine Translation), which contains paired sequences from sources like European Parliament proceedings.
- **Question answering:** The conversational interface you see in ChatGPT, Gemini, and similar assistants. The user asks a question; the model generates a response.
- **Summarization:** Condense a long article into a shorter summary.
- **Open-ended generation:** Generate code, poetry, stories, or any other text.

**Evaluation metrics for generation tasks are harder** because there are many valid ways to express the same meaning (any bilingual person knows there are multiple correct translations of a sentence):

- **BLEU (Bilingual Evaluation Understudy):** Measures n-gram overlap between the model's output and a reference translation. Higher is better. It counts how many n-grams (unigrams, bigrams, trigrams, etc.) in the generated text appear in the reference.
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** A suite of metrics that focus on recall of n-grams from the reference. Higher is better. Where BLEU emphasizes precision, ROUGE emphasizes recall.
- **Perplexity:** Measures how "surprised" the model is by its own outputs. Formally, it is the exponentiated average negative log-likelihood. Lower is better. A perplexity of 1 means the model is perfectly confident; higher values indicate more uncertainty.

Both BLEU and ROUGE are **reference-based**, meaning they require ground-truth labels, which are expensive to produce. Recent progress in LLMs is enabling **reference-free** evaluation, where a strong LLM acts as a judge.

### Historical context

The field did not start in 2022 with ChatGPT. RNN-like architectures were conceived in the 1980s. LSTMs appeared in the 1990s. Word2Vec (2013) was a breakthrough in learning useful word representations. The Transformer (2017) provided the foundational architecture. The 2020s brought massive scaling of data and compute, producing what we now call LLMs.

---

## 2. Tokenization

Models work with numbers, not raw text. **Tokenization** is the process of breaking text into discrete units called **tokens** that can then be mapped to numerical representations.

Given the sentence "A cute teddy bear is reading", we need to decide how to split it before passing it to a model. There are three main strategies:

### 2.1 Word-level tokenization

Split on whitespace/punctuation. "A", "cute", "teddy", "bear", "is", "reading" each become one token.

**Pros:** Simple and intuitive.

**Cons:**
- Does not leverage word roots. "bear" and "bears" are treated as entirely unrelated tokens, even though they share the same root. The model must independently learn that these are related.
- High **OOV (out-of-vocabulary) risk.** At inference time, if the model encounters a word it never saw during training, it must map it to a generic `<UNK>` (unknown) token, losing all information about that word. With word-level tokenization, this happens frequently because every inflected form (runs, running, ran) is a separate vocabulary entry.

### 2.2 Subword-level tokenization

Split words into meaningful sub-units based on commonly occurring character sequences. For example, "bears" becomes "bear" + "s", and "running" becomes "run" + "ning". This means "bear" and "bears" share the "bear" token.

**Pros:**
- Leverages morphological roots, so related words share subword tokens.
- Much lower OOV risk, since rare words can be decomposed into known subwords.

**Cons:**
- Produces longer token sequences than word-level (e.g., one word might become 2-3 tokens). This matters because the computational complexity of transformer models grows with sequence length (specifically, self-attention is $O(n^2)$ in the sequence length $n$).

**This is the standard approach used in practice today** (e.g., BPE, WordPiece, SentencePiece).

### 2.3 Character-level tokenization

Every individual character is a token: "A", " ", "c", "u", "t", "e", ...

**Pros:**
- Robust to misspellings and casing variations, since every character is in the vocabulary.
- Essentially zero OOV risk.

**Cons:**
- Sequences become very long (a 10-word sentence might be 50+ characters), dramatically increasing computation time.
- It is hard to learn meaningful representations of individual characters in isolation. What does the embedding of the letter "u" mean by itself?

### Summary

| Level | Root leverage | OOV risk | Sequence length |
|---|---|---|---|
| Word | None | High | Short |
| Subword | Yes | Low | Medium |
| Character | N/A | ~Zero | Very long |

**Vocabulary sizes in practice:** Roughly tens of thousands of tokens for a single-language model; hundreds of thousands for multilingual models that also handle code. For languages with non-Latin scripts (e.g., Chinese, Japanese), the vocabulary includes those characters as subword units.

**Special tokens:** Sequences typically include special tokens like `<BOS>` (beginning of sequence) and `<EOS>` (end of sequence) to mark boundaries. During generation, the model stops producing tokens when it emits the `<EOS>` token.

---

## 3. Word/Token Representation

Once we have tokens, we need to represent each one as a numerical vector that a model can process. The quality of these representations directly determines how well the model can reason about language.

### 3.1 One-hot encoding

The simplest approach: assign each token a vector of length $V$ (vocabulary size) with a 1 in the position corresponding to that token and 0s everywhere else.

For a vocabulary {soft, teddy bear, book}:
- "soft" $\to [1, 0, 0]$
- "teddy bear" $\to [0, 1, 0]$
- "book" $\to [0, 0, 1]$

**The fundamental problem:** We want to measure how similar two tokens are. The standard way to do this is cosine similarity:

$$\text{cosine similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \, \|\mathbf{v}\|}$$

This measures the angle between two vectors. Vectors pointing in the same direction have similarity 1; orthogonal vectors have similarity 0; opposite vectors have similarity -1.

With one-hot vectors, every pair of distinct tokens is orthogonal (their dot product is always 0). The model has no way to know that "teddy bear" and "soft" are semantically related while "teddy bear" and "book" are less so. Every token is equally distant from every other token.

**What we actually want:** Tokens with similar meanings should have high cosine similarity, and unrelated tokens should have similarity near 0.

### 3.2 Word2Vec (2013)

Word2Vec was a landmark paper that showed how to **learn dense, low-dimensional embeddings** from large amounts of text using a simple neural network and a **proxy task**.

The key insight: we do not actually care about performing the proxy task well. We care about the representations the model learns along the way. If a model can predict surrounding words from a target word (or vice versa), then its internal representations must encode something meaningful about language.

**Two training approaches:**
- **CBOW (Continuous Bag of Words):** Given the surrounding context words (e.g., "a", "teddy", "is", "reading"), predict the target word ("cute"). The model learns which words tend to appear in similar contexts.
- **Skip-gram:** The reverse. Given a target word ("cute"), predict the surrounding context words. Skip-gram tends to work better for rare words.

**Architecture:** A shallow neural network with one hidden layer:

$$\mathbf{h} = W_1 \mathbf{x} + \mathbf{b}_1, \quad \hat{\mathbf{y}} = \text{softmax}(W_2 \mathbf{h} + \mathbf{b}_2)$$

where:
- $\mathbf{x} \in \mathbb{R}^V$ is the one-hot input vector
- $W_1 \in \mathbb{R}^{D \times V}$ projects from vocabulary space to embedding space
- $W_2 \in \mathbb{R}^{V \times D}$ projects back to vocabulary space for prediction
- $D$ is the embedding dimension (typically hundreds, e.g., 768), with $D \ll V$

**How training works step by step:**

1. Take a word from the corpus (e.g., "A") and represent it as a one-hot vector.
2. Multiply by $W_1$ to get the hidden representation $\mathbf{h} \in \mathbb{R}^D$ (e.g., a vector $[2, 1.9]$ if $D=2$).
3. Multiply by $W_2$ and apply softmax to get a probability distribution over the vocabulary: the model's prediction of what the next word is.
4. Compare the prediction to the true next word using **cross-entropy loss**, which measures how far off the prediction is from the ground truth.
5. **Backpropagate** the error to update $W_1$ and $W_2$, nudging the model's predictions closer to reality.
6. Repeat for every word in the corpus, for multiple passes (**epochs**), until the loss converges.

**After training:** The weight matrix $W_1$ contains the learned embeddings. To get the embedding of any word, take its one-hot vector and multiply by $W_1$. The resulting $D$-dimensional vector is a dense, meaningful representation of that word.

**Famous result:** The learned embeddings capture semantic relationships through vector arithmetic:

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

$$\vec{\text{Paris}} - \vec{\text{France}} + \vec{\text{Germany}} \approx \vec{\text{Berlin}}$$

This showed that the embedding space has geometric structure reflecting real-world relationships.

**Key limitation:** Each word gets exactly one embedding regardless of context. The word "bank" has the same vector whether it appears in "river bank" or "bank robbery". This is a fundamental problem that later architectures (RNNs and Transformers) address by producing **context-dependent** representations.

**When to stop training:** Track the loss function across epochs. When it converges (stops decreasing meaningfully), the embeddings have stabilized and training can stop. The downstream task may also inform when representations are "good enough".

**Choosing the embedding dimension $D$:** This is a trade-off. A larger $D$ gives the model more capacity to encode nuanced relationships, which helps for complex downstream tasks. But larger vectors mean more computation at every subsequent step (more multiply-adds, higher memory usage, slower inference). In practice, values like 256, 512, or 768 are common.

---

## 4. Recurrent Neural Networks (RNNs)

Word2Vec gives us embeddings for individual tokens, but we also need to represent **sequences**. A naive approach -- averaging all the word vectors in a sentence -- loses word order entirely ("dog bites man" and "man bites dog" would get the same representation).

RNNs address this by processing tokens **one at a time in order**, maintaining a **hidden state** vector that serves as a running summary of the sequence seen so far. The architecture was first conceived in the 1980s.

### How RNNs work

At each time step $t$, the RNN takes two inputs:
1. The current token's embedding $\mathbf{x}^{(t)}$
2. The hidden state from the previous step $\mathbf{a}^{(t-1)}$ (initialized to zeros or a learned vector at $t=0$)

It produces an updated hidden state:

$$\mathbf{a}^{(t)} = f\!\left(W_a \mathbf{a}^{(t-1)} + W_x \mathbf{x}^{(t)} + \mathbf{b}\right)$$

where $f$ is a nonlinear activation function (e.g., tanh), $W_a$ and $W_x$ are learned weight matrices, and $\mathbf{b}$ is a bias term. The same weights are shared across all time steps (this is what makes it "recurrent").

You can think of $\mathbf{a}^{(t)}$ as the model's "memory" of the sentence up to position $t$. It compresses all the information from tokens $1$ through $t$ into a single fixed-size vector.

### Using RNNs for different NLP tasks

- **Classification:** Process the entire sequence, then use the final hidden state $\mathbf{a}^{(T)}$ (which summarizes the whole sentence) and project it into label space through a linear layer + softmax.
- **Multi-classification (e.g., NER):** Use each hidden state $\mathbf{a}^{(t)}$ to predict the label for the corresponding token $t$.
- **Generation (e.g., translation):** Process the entire source sentence through the encoder to get a final context vector $\mathbf{a}^{(T)}$. Then use this vector to initialize a decoder RNN that generates the target sequence one token at a time.

### The vanishing gradient problem

RNNs struggle with long sequences. To understand why, consider what happens during backpropagation.

To update the weights at the beginning of a long sequence based on a loss computed at the end, the gradient must flow backward through every time step. This involves a chain of multiplications:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(T)}} \cdot \prod_{t=2}^{T} \frac{\partial \mathbf{a}^{(t)}}{\partial \mathbf{a}^{(t-1)}}$$

Each factor $\frac{\partial \mathbf{a}^{(t)}}{\partial \mathbf{a}^{(t-1)}}$ depends on the weights and activation function. If these factors are consistently less than 1, the product shrinks exponentially toward 0 (the gradient **vanishes**), making early tokens essentially invisible to the training signal. If the factors are consistently greater than 1, the product explodes toward infinity (the gradient **explodes**), causing unstable training. This is called **backpropagation through time (BPTT)**.

The practical consequence: RNNs cannot effectively learn dependencies between tokens that are far apart in a sequence. This is referred to as the **long-range dependency** problem.

### Additional problem: slow training

Because each time step depends on the previous one ($\mathbf{a}^{(t)}$ requires $\mathbf{a}^{(t-1)}$), RNNs cannot be parallelized across the sequence dimension. You must compute step 1 before step 2 before step 3, etc. This makes training on long sequences very slow compared to architectures that can process all positions simultaneously.

### 4.1 LSTMs (Long Short-Term Memory, 1997)

LSTMs are an extension of RNNs designed to mitigate the vanishing gradient problem. They introduce a **cell state** $\mathbf{c}^{(t)}$ in addition to the hidden state $\mathbf{a}^{(t)}$.

The cell state acts as a "conveyor belt" that can carry information across many time steps with minimal modification. The LSTM uses learned **gates** (forget gate, input gate, output gate) to decide what information to add to, remove from, or read from the cell state at each step.

This architecture helps the model retain important information over longer sequences. However, it does not fully solve the long-range dependency problem for very long sequences, and it still cannot be parallelized -- each step still depends on the previous one.

---

## 5. Attention Mechanism (2014)

The attention mechanism was introduced to address the information bottleneck in RNN-based sequence-to-sequence models (Bahdanau et al., 2014).

**The core problem it solves:** In a standard encoder-decoder RNN for translation, the entire source sentence must be compressed into a single fixed-size context vector $\mathbf{a}^{(T)}$. For long sentences, this vector cannot capture everything, and information about early tokens gets lost.

**The solution:** Instead of relying solely on the final hidden state, let the decoder look at **all** encoder hidden states at every decoding step. At each step, the decoder computes a set of **attention weights** over the encoder states -- essentially deciding "which parts of the input should I focus on right now?" -- and takes a weighted sum.

For example, when translating "A cute teddy bear is reading" to French and generating the word "mignon" (cute), the attention mechanism assigns high weight to the encoder state corresponding to "cute" and low weights to other positions. This creates a **direct link** between the output token being generated and the relevant input tokens, bypassing the sequential bottleneck.

This was a major conceptual shift: the model no longer has to compress everything into one vector. It can "peek" at any part of the input at any time.

---

## 6. The Transformer (2017)

**Paper:** "Attention Is All You Need" (Vaswani et al., 2017)

The Transformer took the attention idea to its logical conclusion: **remove the recurrence entirely** and rely solely on attention to model relationships between tokens. This architecture is the foundation of all modern LLMs (GPT, BERT, T5, LLaMA, etc.).

The key innovation is **self-attention**: instead of attending from decoder to encoder (as in the 2014 attention mechanism), tokens attend to **each other** within the same sequence. This allows the model to compute a context-aware representation of each token as a function of all other tokens in the sequence -- simultaneously, in parallel.

Going back to the "bank" example: with self-attention, the representation of "bank" in "river bank" would be different from "bank" in "bank robbery", because the surrounding tokens that it attends to are different.

### 6.1 Self-Attention with Query, Key, Value

The self-attention mechanism uses three learned projections to transform the input embeddings:

Given input embeddings $X \in \mathbb{R}^{n \times d_{\text{model}}}$ (one row per token):

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ are learned projection matrices.

**Intuition behind Q, K, V:**
- **Query** ($Q$): "What am I looking for?" Each token's query represents the information it wants to gather from the rest of the sequence.
- **Key** ($K$): "What do I contain?" Each token's key is what gets compared against queries to determine relevance.
- **Value** ($V$): "What information do I provide?" Once we know which tokens are relevant (via Q-K matching), the value is the actual content we extract.

Think of it like a search engine: the query is your search term, the key is the title of each document, and the value is the document content. You match queries to keys to find relevant documents, then read the values of the matching ones.

**The self-attention formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Step by step:**

1. **Compute similarity scores:** $QK^\top \in \mathbb{R}^{n \times n}$. Each entry $(i, j)$ is the dot product between query $i$ and key $j$, measuring how much token $i$ should attend to token $j$. This produces a full $n \times n$ matrix of pairwise attention scores.

2. **Scale:** Divide by $\sqrt{d_k}$. As $d_k$ grows, dot products tend to grow in magnitude, pushing softmax into regions where its gradients are extremely small (saturation). Scaling by $\sqrt{d_k}$ keeps the values in a range where softmax gradients are healthy.

3. **Normalize:** Apply softmax row-wise. Each row becomes a probability distribution summing to 1, representing how much each token attends to every other token.

4. **Aggregate:** Multiply by $V$. Each row of the output is a weighted sum of all value vectors, where the weights come from the attention distribution. Tokens that were deemed more relevant (higher attention weight) contribute more to the output representation.

The result is a new set of embeddings where each token's representation is **context-aware** -- it incorporates information from all other tokens in the sequence, weighted by relevance.

**Why this enables parallelism:** Unlike RNNs, there are no sequential dependencies. The entire $QK^\top$ matrix can be computed in one batched matrix multiplication, and GPUs are optimized for exactly this kind of operation. This is what makes Transformers dramatically faster to train than RNNs.

### 6.2 Multi-Head Attention

A single attention computation captures one type of relationship between tokens. But language has many simultaneous relationships (syntactic, semantic, positional, etc.). **Multi-head attention** runs the self-attention computation $h$ times in parallel, each with its own set of learned projection matrices:

$$\text{head}_i = \text{Attention}(X W_Q^{(i)},\; X W_K^{(i)},\; X W_V^{(i)})$$

Each head can learn to focus on different types of patterns. For example, one head might learn syntactic dependencies (subject-verb agreement), another might learn semantic similarity, and another might learn positional patterns.

The outputs of all heads are concatenated and projected back to the model dimension:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \, W_O$$

where $W_O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$ ensures the output has the same dimension as the input, making layers stackable.

This is analogous to using multiple filters in a convolutional neural network: each filter detects a different type of pattern, and combining them gives a richer representation than any single filter could.

**Why do different heads learn different things?** There is no explicit constraint forcing diversity. However, gradient descent naturally drives each head toward learning a different projection because duplicating the same computation provides no benefit to the loss function -- the model has an incentive to use its capacity efficiently.

### 6.3 Architecture: Encoder-Decoder

The original Transformer uses an encoder-decoder structure, designed for sequence-to-sequence tasks like machine translation.

#### Encoder (processes the input/source text)

The encoder's job is to produce rich, context-aware representations of every token in the input sequence.

1. **Input embedding + positional encoding:** Each token is mapped to a $d_{\text{model}}$-dimensional embedding. Since self-attention has no inherent notion of word order (unlike RNNs), **positional encodings** are added element-wise to the embeddings to inject position information. The original paper uses sinusoidal functions of different frequencies for this.

2. **Multi-head self-attention:** All input tokens attend to all other input tokens. After this layer, each token's representation incorporates information from the entire input sequence.

3. **Feed-forward network (FFN):** A position-wise fully connected network applied independently to each token:

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$$

The inner dimension $d_{ff}$ is typically larger than $d_{\text{model}}$ (e.g., $d_{ff} = 4 \times d_{\text{model}}$). This expansion gives the model additional capacity to learn complex transformations before projecting back down. You can think of it as: attention mixes information across tokens, and the FFN processes each token's mixed representation in a richer space.

4. **Stack $N$ times:** The encoder consists of $N$ identical layers stacked on top of each other (the original paper uses $N=6$). Each layer refines the representations further.

5. **Output:** The final encoder layer produces a matrix of context-aware embeddings, one per input token.

#### Decoder (generates the output/target text)

The decoder generates the output sequence one token at a time, using both the encoded input and the tokens generated so far.

1. **Start with BOS token:** Generation begins with the `<BOS>` (beginning of sequence) special token.

2. **Masked self-attention:** The decoder tokens attend to each other, but only to tokens at **earlier** positions (i.e., tokens that have already been generated). This is enforced by **masking**: setting attention scores for future positions to $-\infty$ before softmax, so they get zero weight. This prevents the model from "cheating" by looking at tokens it has not generated yet.

3. **Cross-attention:** This is where the decoder "reads" the input. The **queries** come from the decoder's current state (representing "what am I trying to generate next?"), while the **keys and values** come from the encoder's output (representing the input sentence). This lets the decoder focus on the relevant parts of the input for each output token it generates.

4. **Feed-forward network:** Same structure as in the encoder.

5. **Stack $N$ times:** The decoder also consists of $N$ stacked layers. The cross-attention in every decoder layer connects to the **same** final encoder output.

6. **Output prediction:** After the final decoder layer, a linear projection maps the representation to a vector of size $V$ (vocabulary size), and softmax converts it to a probability distribution:

$$P(\text{token}_i) = \text{softmax}(W_{\text{out}} \mathbf{h} + \mathbf{b})_i$$

The token with the highest probability (or sampled from the distribution) becomes the next generated token.

7. **Repeat:** The newly generated token is fed back into the decoder as input, and the process repeats until the model generates the `<EOS>` (end of sequence) token.

#### Summary of the three attention types

| Attention type | Where | Queries from | Keys/Values from | Purpose |
|---|---|---|---|---|
| Self-attention | Encoder | Input tokens | Input tokens | Build context-aware input representations |
| Masked self-attention | Decoder | Output tokens so far | Output tokens so far | Represent decoded sequence without looking ahead |
| Cross-attention | Decoder | Decoder states | Encoder output | Let decoder focus on relevant input tokens |

### 6.4 Label Smoothing

A training technique used in the original Transformer paper. In NLP, there is often more than one correct next word ("What a great ___" could be "day", "lecture", "idea", etc.). Standard cross-entropy training uses hard targets -- a 1 for the correct token and 0 for everything else -- which pushes the model toward absolute certainty.

**Label smoothing** softens these targets:

$$y_i = \begin{cases} 1 - \varepsilon & \text{if } i = \text{target} \\ \frac{\varepsilon}{V - 1} & \text{otherwise} \end{cases}$$

where $\varepsilon$ is a small constant (e.g., 0.1). Instead of training the model to assign 100% probability to one token, we train it to assign $(1 - \varepsilon)$ probability and distribute the remaining $\varepsilon$ uniformly across all other tokens.

**Effect:** The model becomes less overconfident in its predictions, which acts as a form of regularization. The authors found that label smoothing empirically improves BLEU scores for translation, even though it slightly increases perplexity (the model becomes intentionally less "sure").

---

## Key Takeaways

1. **Tokenization** is the first step in any NLP pipeline. Subword tokenization is the practical standard, balancing vocabulary size, OOV risk, and sequence length.
2. **One-hot encoding** fails because it cannot express similarity between tokens. **Word2Vec** showed that dense embeddings learned from a proxy task (next-word prediction) capture rich semantic relationships, but each word gets only one context-independent vector.
3. **RNNs** introduced sequential processing and word-order awareness, but suffer from vanishing gradients (cannot learn long-range dependencies) and are inherently sequential (slow to train). **LSTMs** partially mitigate the gradient problem with a cell state mechanism but remain sequential.
4. **Attention** (2014) created direct connections between distant tokens, bypassing the sequential bottleneck. The decoder can "peek" at any part of the input at each generation step.
5. **The Transformer** (2017) removed recurrence entirely, relying solely on self-attention. It processes all tokens in parallel via matrix operations that GPUs handle efficiently, and produces context-aware representations where each token is informed by every other token.
6. The **encoder-decoder** structure with three attention types (self, masked self, cross) enables sequence-to-sequence tasks like machine translation. Training techniques like **label smoothing** improve generation quality by preventing overconfidence.
