# Lecture 6: LLM Reasoning - Stanford CME 295

**Date:** November 7, 2025
**Instructors:** Afshine Amidi & Shervine Amidi (Adjunct Lecturers, Stanford)
**Course:** CME 295 - Transformers and Large Language Models, Autumn 2025

---

## 1. Recap: The Training Pipeline So Far

Before introducing reasoning, the lecture situates where we stand in the three-stage training pipeline covered in lectures 4 and 5.

### 1.1 Pre-training (Lecture 4)

The most compute-intensive step. The model learns the structure of text and code through large-scale next-token prediction on massive corpora. At the end of pre-training, the model can autocomplete sequences but has no ability to follow instructions or behave as a useful assistant.

### 1.2 Supervised Fine-Tuning (Lecture 4)

The pre-trained model is fine-tuned on high-quality instruction-response pairs (called SFT data) so that it learns to respond helpfully to user queries. After SFT, the model can act as a task-specific assistant (e.g., answering questions).

### 1.3 Preference Tuning with RLHF (Lecture 5)

The SFT model is further aligned to human preferences using reinforcement learning from human feedback (RLHF). This stage has two components: first, a reward model is trained to distinguish good from bad completions using human preference data; second, an RL stage tunes the policy (the LLM) to maximize rewards while staying close to the reference model.

The RL loss function has two parts:

1. **Advantage maximization** -- The model is pushed to increase the probability of generating tokens that lead to high-reward completions. The advantage measures how much better a completion is compared to some baseline.
2. **Regularization via KL divergence** -- The model is penalized for deviating too far from the reference model (the SFT checkpoint). This prevents catastrophic forgetting of the capabilities already learned.

The standard RL algorithm for this stage is **PPO (Proximal Policy Optimization)**, which uses a clipping mechanism to keep policy updates within a bounded region:

$$L^{\text{PPO}} = \mathbb{E}\left[\min\left(r_t \hat{A}_t, \; \text{clip}(r_t, 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right)\right]$$

where $r_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the probability ratio between the current and old policy, and $\hat{A}_t$ is the estimated advantage. The clip function prevents the ratio from exceeding $1 \pm \epsilon$, ensuring no single update is too large.

There is also a **PPO-KL penalty** variant that explicitly adds a KL divergence term to penalize deviation from the base (SFT) model. In modern RLHF training, practitioners typically combine elements of both variants.

---

## 2. Strengths and Weaknesses of Vanilla LLMs

Everything covered in lectures 1 through 5 produces what this lecture calls "vanilla LLMs" -- models that take a prompt and produce a direct answer. These models have notable strengths and limitations.

### 2.1 Strengths

Vanilla LLMs are excellent at tasks they absorbed from pre-training: debugging code, generating code, writing essays, producing creative text like poetry, and answering factual questions that appeared in their training data.

### 2.2 Weaknesses

Four weaknesses motivate the rest of the lecture and the next two lectures in the course:

1. **Limited reasoning** -- Vanilla LLMs struggle with sophisticated multi-step problems (e.g., math olympiad questions). They are trained to produce plausible next tokens, not to decompose and methodically solve hard problems. There is no inherent mechanism forcing multi-step deliberation.

2. **Static knowledge (cutoff date)** -- The model's knowledge is bounded by the pre-training data cutoff. Events after that date (e.g., election results) are unknown to the model. This limitation is addressed in Lecture 7.

3. **No ability to take actions** -- Vanilla LLMs can only produce text. They cannot place orders, call APIs, or interact with external systems. This is addressed in Lecture 8.

4. **Difficult evaluation** -- Unlike traditional NLP tasks that used reference-based metrics (BLEU for translation, ROUGE for summarization), LLMs produce free-form text that is hard to evaluate with rule-based metrics. This is an ongoing challenge.

The focus of this lecture is the first weakness: reasoning.

---

## 3. What Is Reasoning?

There is no universally agreed-upon definition of reasoning in the context of LLMs. The working definition used in this lecture:

**Reasoning is the ability to solve a problem through a multi-step process** -- decomposing it into tractable subproblems, working through them sequentially, and arriving at a final answer. The kinds of problems that require reasoning are typically mathematical or algorithmic, though the hope is that these capabilities generalize to other domains.

### 3.1 Reasoning vs. Knowledge Retrieval

A non-reasoning question asks for a stored fact: "What is the course code of Stanford's Transformers and LLM class?" The answer (CME 295) is a knowledge lookup.

A reasoning question requires computation across multiple steps: "A bear was born in 2020. How old is the bear in 2025?" This is trivial, but the same structure scales to problems like competition mathematics where direct pattern matching from training data is insufficient.

### 3.2 From Vanilla LLMs to Reasoning Models

The output of a vanilla LLM is just an answer. The output of a reasoning model is a **reasoning chain followed by an answer**. The model first generates an extended sequence of intermediate thinking steps (the reasoning chain), and only then produces the final answer.

This distinction is visible in commercial products. When using ChatGPT, Gemini, or Claude in reasoning mode, the UI displays a "Thinking..." indicator while the model generates its reasoning chain. The user typically sees a summary of the reasoning, not the raw chain. There are several reasons for this: the raw chain may not be fully intelligible to humans; users may not want to read pages of intermediate reasoning; and exposing raw reasoning chains would allow competitors to train on them.

From a pricing perspective, reasoning tokens count as output tokens. Users are charged for both the reasoning chain and the final answer, which creates an economic incentive to maximize reasoning quality while minimizing reasoning length.

### 3.3 Timeline of Reasoning Models

Reasoning as a research topic predates 2024, but reasoning models as deployable products began with OpenAI's release of **o1-preview in September 2024**. This triggered a wave of activity across the industry:

- **December 2024**: Google's Gemini 2.0 Flash Thinking
- **January 2025**: DeepSeek R1 -- a landmark moment because DeepSeek matched OpenAI's reasoning performance and published their methodology openly
- **2025 onward**: xAI, Anthropic (Claude), Mistral, and others added reasoning capabilities to their models

Nearly everything covered in this lecture is from 2024 or 2025.

---

## 4. Chain of Thought at Scale

### 4.1 The Core Idea

The mechanism behind reasoning models is an extension of **chain-of-thought (CoT) prompting**, a technique introduced earlier in the course. In CoT prompting, the user provides in-context examples that explicitly show step-by-step reasoning, encouraging the model to follow the same pattern before giving its answer.

Reasoning models apply this idea at a much larger scale: instead of relying on prompting tricks, the model is trained to always produce extended reasoning chains before answering.

### 4.2 Why More Tokens Help

There are two complementary intuitions for why generating a reasoning chain improves performance:

1. **Decomposition into tractable subproblems** -- Hard problems are unlikely to have appeared verbatim in the training data. By decomposing a hard problem into simpler steps, the model can match each step against patterns it did learn during pre-training. This mirrors how a student solves an exam problem by linking it back to techniques studied in class.

2. **More compute per problem** -- Each generated token requires a full forward pass through the network. Generating more tokens before answering literally gives the model more computation to work with. This connects to the concept of **compute budgets** -- the amount of inference-time computation allocated to a problem.

### 4.3 Controlling the Thinking Budget

Not all prompts require the same amount of reasoning. A simple factual question should not trigger pages of deliberation. Several approaches exist to control how much the model thinks:

- **Dynamic budget classification** -- A lightweight classifier runs on the prompt first to categorize it as high-thinking or low-thinking, and the reasoning budget is set accordingly.

- **Budget forcing** (introduced in the S1 paper) -- Special tokens are injected into the reasoning chain to either extend or terminate thinking. For example, inserting "Wait, I think there is another solution..." forces the model to explore an additional reasoning path. Conversely, inserting "Okay, your time is up. My answer is..." forces the model to conclude.

- **Continuous thought representations** -- Instead of reasoning in the token (language) space, some research explores having the model reason in a continuous hidden-state space. The "thinking tokens" become hidden representations rather than discrete words, which can be more compressed and expressive. This is an active area of research with papers appearing as recently as late 2025.

---

## 5. Reasoning Benchmarks

### 5.1 Coding Benchmarks

The setup: given a coding problem, the model must produce a solution that passes all test cases. Correctness is binary and deterministically verifiable -- the code either passes every test case or it does not.

Key benchmarks include:

- **HumanEval** -- Approximately 164 hand-written Python programming problems
- **Codeforces** -- Problems drawn from competitive programming contests
- **SWE-bench** -- Problems derived from real GitHub issues, testing practical software engineering skills

### 5.2 Math Benchmarks

The setup: given a math problem, the model must produce the answer. Verification is done by parsing the answer (e.g., from a boxed expression format) and comparing it to a ground truth value.

Key benchmarks include:

- **AIME (American Invitational Mathematics Examination)** -- Competition-level problems used to qualify for the US Math Olympiad. This is one of the most commonly cited reasoning benchmarks.
- **GSM8K** -- Grade-school-level math problems (simpler than AIME, used as a lower bar).

### 5.3 The Pass@k Metric

The primary metric for reasoning benchmarks is **pass@k**, defined as the probability that at least one of $k$ sampled attempts produces a correct answer.

The motivation: in settings where correctness is verifiable (code that passes tests, math with known answers), it can be worthwhile to generate multiple candidate answers and check each one. If any single candidate is correct, the problem is solved. This is analogous to the **best-of-n** sampling technique from preference tuning (Lecture 5), except here we use a deterministic verifier rather than a learned reward model.

#### Derivation of the pass@k estimator

To estimate pass@k, we generate $n$ total samples (where $n \geq k$) and observe that $c$ of them are correct and $n - c$ are incorrect. We want to compute: out of $n$ total samples, if we randomly select $k$, what is the probability that at least one is correct?

Using the complement rule:

$$\text{pass@}k = 1 - P(\text{all } k \text{ attempts incorrect})$$

The probability of drawing $k$ incorrect samples without replacement from a pool of $n$ samples (of which $n-c$ are incorrect) is:

$$P(\text{all } k \text{ incorrect}) = \frac{n-c}{n} \cdot \frac{n-c-1}{n-1} \cdot \frac{n-c-2}{n-2} \cdots \frac{n-c-k+1}{n-k+1}$$

This is sampling without replacement. Using combinatorial notation:

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient. This formula counts the number of ways to choose $k$ incorrect samples out of $n-c$ incorrect ones, divided by the total number of ways to choose $k$ samples out of $n$.

**Special case -- pass@1:** Substituting $k=1$:

$$\text{pass@}1 = 1 - \frac{n-c}{n} = \frac{c}{n}$$

This simplifies to the proportion of successful attempts, which matches the intuitive definition of accuracy.

**Why not just generate exactly $k$ samples?** Because the resulting estimate would be very noisy. By generating $n \gg k$ samples and using the combinatorial formula, the estimate has much lower variance.

### 5.4 Temperature and Pass@k

Temperature controls the diversity of generated solutions and directly affects pass@k performance:

- **Very low temperature (e.g., $T = 0$)**: The model produces nearly identical outputs every time. All $k$ samples are effectively the same, so generating more does not improve the probability of finding a correct one. Pass@k stays flat as $k$ increases.

- **Moderate temperature (e.g., $T = 0.4$ to $T = 0.8$)**: Solutions are diverse enough that additional samples explore different reasoning paths, but the temperature is not so high as to degrade quality. Pass@k improves meaningfully with $k$.

- **Very high temperature (e.g., $T = 1.2$)**: Solutions are highly diverse, but quality drops because unlikely tokens become probable. This hurts performance, especially at low $k$.

The optimal temperature depends on $k$. For larger $k$, slightly higher temperatures tend to perform better because diversity matters more. Papers always specify the temperature used for benchmark results.

### 5.5 Other Metrics

- **Consensus@k** -- The answer that appears most frequently among $k$ generated responses. This is closely related to the **self-consistency** technique, where you sample multiple reasoning chains and take a majority vote on the final answer.
- **Standard metrics** -- Accuracy, exact match, and other familiar metrics are also reported alongside pass@k.

---

## 6. Why RL for Reasoning (Not SFT)

Given the goal of training a model to produce reasoning chains before answering, one might consider supervised fine-tuning on reasoning chain data. Three facts argue against starting with SFT and in favor of RL:

1. **Reasoning chains are expensive to write.** High-quality SFT requires human-authored demonstrations. Writing detailed, multi-step reasoning chains for thousands of problems is extremely labor-intensive, especially for hard problems where the chains can be very long.

2. **The model may reason differently than humans.** Human-written reasoning chains reflect how humans think, but the optimal reasoning strategy for an LLM may be different. Forcing the model to mimic human-style reasoning may be suboptimal.

3. **Reasoning tasks have verifiable rewards.** For coding, correctness is checked by running test cases. For math, correctness is checked by comparing the parsed answer to a ground truth. This means we have a reliable, automatic reward signal -- exactly what RL needs to function without a learned reward model.

The combination of these facts points strongly toward RL: we have a reward signal, we do not have high-quality training data, and we want to let the model discover its own reasoning style.

### 6.1 The Reward Function for Reasoning

The RL reward for reasoning combines two components:

1. **Format reward** -- Checks whether the model produced a properly structured reasoning chain, verified by the presence of designated think-start and think-end tokens (e.g., `<think>` and `</think>`).

2. **Correctness reward** -- Checks whether the final answer is correct. For code: the solution passes all test cases. For math: the parsed answer matches the ground truth.

Both rewards are **verifiable** -- no learned reward model is needed. This is a significant simplification over standard RLHF, where a separate reward model must be trained on human preference data.

### 6.2 Empirical Evidence: RL Works

When a pre-trained model is trained with RL using only these two verifiable rewards, performance on reasoning benchmarks (e.g., AIME) increases substantially over the course of training. The DeepSeek R1-Zero experiment (discussed in detail in Section 9) demonstrated this directly: a model with no prior supervised fine-tuning on reasoning data learned to reason through RL alone.

---

## 7. GRPO: Group Relative Policy Optimization

GRPO (Group Relative Policy Optimization), published in 2024, is the dominant RL algorithm for training reasoning models. Like PPO, it maximizes advantages while constraining policy updates. The critical difference is in how it computes advantages.

### 7.1 The Problem with PPO's Advantage Estimation

PPO computes advantages using a **value function** -- a separate neural network trained jointly with the policy. The value function predicts the expected future reward at each token position, and the advantage is the difference between the actual reward and this prediction (computed via Generalized Advantage Estimation, a complex recursive formula).

The problem: training a value function jointly with the policy is expensive. The value network can be as large as the policy network itself, effectively doubling the memory and compute requirements.

### 7.2 GRPO's Solution: Group-Based Advantages

GRPO eliminates the value function entirely. Instead, it estimates advantages by comparing a completion's reward against the rewards of other completions generated for the same prompt.

For a given prompt $q$, GRPO generates $G$ completions $\{o_1, o_2, \ldots, o_G\}$, scores each with the reward function to get rewards $\{r_1, r_2, \ldots, r_G\}$, and computes the advantage of completion $i$ as:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

where $\text{mean}$ and $\text{std}$ are computed over the group of $G$ completions for that same prompt.

This normalization serves an important purpose: it contextualizes rewards relative to problem difficulty. A correct answer to a hard problem (where most completions fail) yields a high advantage because $r_i$ is much larger than the group mean. A correct answer to an easy problem (where most completions succeed) yields a modest advantage because the mean is already high. This implicitly encourages the model to focus its learning on hard problems.

### 7.3 GRPO Training Procedure (Step by Step)

1. Sample a prompt $q$ from the training set.
2. Generate $G$ completions from the current policy $\pi_\theta$.
3. Score each completion with the verifiable reward function to get $G$ rewards.
4. Compute group-normalized advantages for each completion.
5. Update the policy to increase the probability of high-advantage completions and decrease the probability of low-advantage ones, subject to a clipping constraint that prevents updates from being too large.
6. Apply a KL divergence penalty to keep the policy close to the reference model (the SFT or pre-trained checkpoint).

### 7.4 GRPO vs. PPO: Side-by-Side Comparison

| Aspect | PPO | GRPO |
|---|---|---|
| Completions per prompt | 1 | $G$ (multiple) |
| Advantage estimation | Reward + value function (GAE) | Group-normalized rewards |
| Value function required | Yes (trained jointly) | No |
| KL divergence | Typically folded into per-token rewards | Explicit term in the objective |
| Primary use case | Preference tuning (RLHF) | Reasoning training |
| Models trained | Policy + value network | Policy only |

In both algorithms, the reward model is frozen (not trained during RL). In the reasoning case specifically, there is no learned reward model at all -- only verifiable reward functions.

### 7.5 The GRPO Loss Function

The full GRPO objective is:

$$J_{\text{GRPO}} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\left(r_{i,t} \hat{A}_i, \; \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i\right) \right] - \beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

where:
- $r_{i,t} = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}$ is the probability ratio for token $t$ in completion $i$
- $\hat{A}_i$ is the group-normalized advantage for completion $i$ (shared across all tokens in that completion)
- $\epsilon$ is the clipping threshold
- $\beta$ is the weight on the KL divergence penalty
- $|o_i|$ is the length of completion $i$
- The outer sum runs over all $G$ completions; the inner sum runs over all tokens in each completion

The clipping mechanism works identically to PPO-clip: it prevents the policy from making excessively large updates in a single iteration.

---

## 8. Extensions and Fixes to GRPO

While GRPO was a major step forward, practitioners quickly identified problems in its original formulation. Three notable issues and their fixes emerged in early 2025.

### 8.1 Length Bias: The $\frac{1}{|o_i|}$ Problem

The most impactful issue in the original GRPO objective is the per-completion normalization factor $\frac{1}{|o_i|}$. To understand the problem, rearrange the loss by distributing this factor inside the inner sum:

$$\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} f(r_{i,t}, \hat{A}_i) = \sum_{t=1}^{|o_i|} \frac{1}{|o_i|} f(r_{i,t}, \hat{A}_i)$$

Now each token's contribution to the objective is scaled by $\frac{1}{|o_i|}$. This means:

- Tokens in **short** completions have **large** per-token weight ($\frac{1}{|o_i|}$ is large when $|o_i|$ is small).
- Tokens in **long** completions have **small** per-token weight ($\frac{1}{|o_i|}$ is small when $|o_i|$ is large).

Consider the case where the advantage is **negative** (a bad completion). The model should downweight the probability of every token in that completion. But with $\frac{1}{|o_i|}$ normalization, tokens in short bad completions are penalized more heavily than tokens in long bad completions. The model learns that a long bad answer is less punished than a short bad answer, creating an incentive to produce longer outputs regardless of quality.

Empirically, this manifests as ever-increasing output lengths during RL training. Plots of average response length vs. RL step show a continuous upward trend. Early in training, this length increase correlates with improving performance (the model is learning to produce more thorough reasoning chains). But eventually, performance plateaus while length continues growing -- the model is generating unnecessarily long outputs.

### 8.2 Fix: DAPO and Dr. GRPO

Two papers from early 2025 proposed fixes:

- **DAPO (published March 2025)** -- Equalizes token-level contributions by replacing $\frac{1}{|o_i|}$ with a normalization factor that is constant across all tokens, regardless of which completion they belong to. This removes the asymmetric treatment of short vs. long completions.

- **Dr. GRPO ("GRPO Done Right")** -- Proposes removing the $\frac{1}{|o_i|}$ normalization factor entirely.

Empirical results confirm that these fixes work: when plotting reward as a function of output length, the corrected algorithms stop the unbounded length increase. In particular, incorrect solutions (negative advantage) become shorter under the corrected policy compared to standard GRPO, while correct solutions maintain appropriate length. This is exactly the intended behavior -- the model no longer pads incorrect answers with unnecessary tokens.

### 8.3 Standard Deviation Bias in Advantage Computation

The group-normalized advantage divides by the standard deviation of rewards within the group. This creates a bias related to problem difficulty. For very hard problems, most completions are incorrect, so the reward distribution has low variance (most rewards are near zero). The small standard deviation amplifies the advantage of any correct completion, which sounds beneficial but can cause instability.

Conversely, a mix of easy and hard problems within a batch can lead to inconsistent gradient magnitudes. Fixes proposed in the literature adjust the normalization to be more robust to skewed reward distributions.

### 8.4 Asymmetric Clipping (Epsilon Bounds)

The standard GRPO clipping bounds are symmetric: $r_{i,t} \in [1 - \epsilon, \; 1 + \epsilon]$. But this means the actual probability change allowed for a token depends on its current probability under the old policy:

$$\pi_\theta(o_{i,t}) \in \left[(1 - \epsilon) \cdot \pi_{\theta_{\text{old}}}(o_{i,t}), \; (1 + \epsilon) \cdot \pi_{\theta_{\text{old}}}(o_{i,t})\right]$$

If $\pi_{\theta_{\text{old}}}(o_{i,t})$ is very small, the absolute range of allowed change is tiny -- the token has almost no room to grow. Meanwhile, tokens with high probability have a large absolute range and could be driven to near-zero probability in one update.

The fix is to use **asymmetric epsilon values**: a larger $\epsilon$ for the lower bound (allowing low-probability tokens more room to grow) and a smaller $\epsilon$ for the upper bound (preventing high-probability tokens from collapsing too quickly). This makes the optimization landscape more balanced across the probability spectrum.

---

## 9. DeepSeek R1-Zero: RL Without Supervision

The DeepSeek R1-Zero experiment (published January 2025) was a proof of concept demonstrating that RL alone, without any supervised fine-tuning on reasoning data, can produce a model with strong reasoning capabilities.

### 9.1 Starting Point

R1-Zero began with **DeepSeek V3 base**, a pre-trained model (next-token prediction only, no SFT) using a modern architecture with mixture-of-experts (MoE), multi-latent attention (MLA, introduced in DeepSeek V2), and RMSNorm in transformer blocks. This model had never seen any supervised instruction data.

### 9.2 Training Setup

RL was applied directly to the pre-trained base model using GRPO with two verifiable rewards:

1. **Correctness reward** -- Whether the final answer is correct (for math/coding problems)
2. **Format reward** -- Whether the output contains properly structured `<think>` blocks followed by an `<answer>` block

The prompt template was simple: a system message instructing the model to place reasoning in `<think>` tags and the final answer in `<answer>` tags, followed by the user's problem.

### 9.3 Results

Performance on AIME (the math benchmark) improved substantially over the course of RL training, demonstrating that a model with no prior instruction tuning can learn to reason just from verifiable reward signals. This was a striking finding: the pre-trained model's latent knowledge of mathematics, combined with the RL incentive to think step-by-step and produce correct answers, was sufficient to develop meaningful reasoning ability.

### 9.4 Limitations of R1-Zero

Despite strong benchmark performance, inspection of R1-Zero's reasoning chains revealed two problems:

1. **Language mixing** -- The model would switch between languages within a single reasoning chain, producing incoherent multilingual output.
2. **Syntax and readability issues** -- The chains had formatting problems and were often difficult for humans to follow.

These issues stem from the absence of any supervised signal to anchor the model's output style. The RL reward only checks correctness and format structure; it does not care about language consistency or readability. This motivated the development of the full R1 pipeline.

---

## 10. DeepSeek R1: The Full Training Pipeline

DeepSeek R1 built on R1-Zero's proof of concept by incorporating supervised stages to address the readability and consistency issues, producing a competitive reasoning model through a multi-stage pipeline.

### 10.1 Stage 1: Cold Start SFT

Starting from the same DeepSeek V3 base model, the first stage uses a small amount of high-quality SFT data. This data was generated by collecting reasoning chains from the R1-Zero stage and having humans rewrite them to fix language mixing, formatting problems, and readability issues. The resulting prompt-response pairs were used for supervised fine-tuning.

The cold start dataset was relatively small -- potentially on the order of thousands of examples, several orders of magnitude less than typical SFT datasets. Its purpose was not to teach reasoning from scratch but to establish proper formatting and language consistency as a foundation for the RL stages.

### 10.2 Stage 2: Reasoning-Focused RL

After cold start SFT, the model underwent RL training (GRPO) on reasoning tasks. The reward function combined three components:

1. **Correctness reward** -- Same as R1-Zero: whether the final answer is right.
2. **Format reward** -- Whether the output uses proper think/answer structure.
3. **Language consistency reward** -- A new addition. This was a simple heuristic measuring the ratio of tokens in the target language within the output chain. By maximizing this ratio, the model was incentivized to avoid the language-mixing problem observed in R1-Zero.

### 10.3 Stage 3: Large-Scale SFT (Rejection Sampling)

After the first RL stage, the model underwent a larger-scale SFT stage mixing two types of data:

- **Non-reasoning data (~200k pairs)** -- Recycled from the DeepSeek V3 instruction-tuning data, covering a broad array of topics (general knowledge, conversation, coding, writing, etc.). This maintains the model's general assistant capabilities.

- **Reasoning data (~600k pairs, 3:1 ratio with non-reasoning)** -- Generated through **rejection sampling**. The process: take reasoning-style prompts, use the current model to generate candidate responses, and filter using automated judges and verifiers. Only high-quality, correct responses with clean reasoning chains are kept. This produces a large-scale SFT dataset of the model's own best outputs.

Rejection sampling is necessary at this scale because human annotation is prohibitively expensive for hundreds of thousands of reasoning chains. Automated filtering (using LLM judges and deterministic verifiers) provides a practical alternative.

### 10.4 Stage 4: Final RL (Reasoning + Non-Reasoning)

The last stage applies RL training with a mixed reward that covers both reasoning and non-reasoning tasks:

- **Reasoning data** -- Same verifiable correctness rewards as before.
- **Non-reasoning data** -- Reward based on two dimensions:
  - **Helpfulness** -- Applied to the user-facing answer portion, measuring how well the model serves as an assistant.
  - **Harmlessness** -- Applied to the entire output including the reasoning chain (not just the final answer), ensuring the model does not produce harmful content even in its internal deliberation.

This final stage aligns the model holistically, producing a model that reasons well on hard problems while remaining helpful and safe on general queries.

### 10.5 Results

DeepSeek R1 was highly competitive with closed-source reasoning models (including OpenAI's o1) across reasoning benchmarks. On non-reasoning benchmarks, it also performed well due to the mixed-data SFT and RL stages. The paper's publication in January 2025 was significant because it demonstrated that a fully described, reproducible training pipeline could match the performance of proprietary reasoning models.

---

## 11. Distillation for Smaller Reasoning Models

Not every deployment can afford a 600-billion-parameter model. The final topic of the lecture addresses how to transfer reasoning capabilities from a large teacher model to a smaller student model.

### 11.1 Distillation in the Reasoning Setting

In pre-training (Lecture 4), distillation meant training a student model to match the teacher's output probability distribution over the next token. In the reasoning setting, the approach is different because we do not have fixed SFT pairs for reasoning.

The distillation process for reasoning models:

1. **Generate data from the teacher** -- Use the large reasoning model (e.g., R1) to produce responses to a large set of prompts, including full reasoning chains with think tokens.
2. **Train the student via SFT** -- Fine-tune the smaller model to predict the exact same token sequence the teacher produced. Rather than matching a probability distribution, the student learns to reproduce the entire reasoning chain and answer.

This is sometimes called **sequence-level distillation**: the student fits the teacher's full output sequence rather than per-token probability distributions.

### 11.2 Distillation vs. RL from Scratch at Small Scale

A natural question: why not just run the same RL pipeline on the smaller model? The DeepSeek authors found that at smaller model sizes, distillation from a strong teacher produces better results than training with RL from scratch. The smaller model does not have enough capacity to discover effective reasoning strategies through RL alone, but it can learn to imitate the teacher's reasoning patterns through supervised training on the teacher's outputs.

### 11.3 Results

The distilled smaller models were competitive with proprietary small reasoning models (e.g., OpenAI's o1-mini) on reasoning benchmarks, demonstrating that the reasoning capabilities of a large model can be effectively compressed into much smaller architectures.

---

## Key Takeaways

1. **Reasoning models** extend vanilla LLMs by generating a multi-step reasoning chain before producing an answer. The output is the reasoning plus the answer, not just the answer. This approach has been mainstream since late 2024.

2. **Chain of thought at scale** is the core mechanism: by generating more intermediate tokens, the model decomposes hard problems into tractable subproblems and allocates more compute per problem. Controlling the amount of reasoning (thinking budgets, budget forcing) remains an active research problem.

3. **Pass@k** is the primary evaluation metric for reasoning tasks, estimating the probability that at least one of $k$ samples is correct: $\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$. Temperature must be tuned for optimal pass@k: too low eliminates diversity, too high degrades quality.

4. **RL with verifiable rewards** is preferred over SFT for training reasoning capabilities because reasoning chains are expensive to write, models may reason differently than humans, and correctness can be checked automatically for math and code.

5. **GRPO** replaces PPO for reasoning training by eliminating the value function and computing advantages through group comparison. For each prompt, $G$ completions are sampled, and each completion's reward is normalized against the group mean and standard deviation.

6. **The length bias in GRPO** arises from per-completion length normalization ($\frac{1}{|o_i|}$), which penalizes short bad answers more than long bad answers, incentivizing ever-longer outputs. DAPO and Dr. GRPO fix this by equalizing or removing the per-token length weighting.

7. **DeepSeek R1-Zero** proved that RL alone (no SFT) can produce reasoning ability from a pre-trained base model, but results in language mixing and readability issues. **DeepSeek R1** solved these issues through a four-stage pipeline: cold start SFT, reasoning RL, large-scale rejection-sampling SFT, and final mixed RL.

8. **Distillation** transfers reasoning capability from a large model to a smaller one by training the student on the teacher's full output sequences (including reasoning chains). At small model sizes, distillation outperforms RL training from scratch.
