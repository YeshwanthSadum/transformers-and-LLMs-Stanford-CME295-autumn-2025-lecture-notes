# Lecture 5: LLM Tuning - Stanford CME 295

**Date:** October 31, 2025
**Instructors:** Afshine Amidi & Shervine Amidi (Adjunct Lecturers, Stanford)
**Course:** CME 295 - Transformers and Large Language Models, Autumn 2025

---

## 1. Recap: The Three-Stage LLM Training Pipeline

Modern LLMs go through a multi-stage training pipeline, and this lecture concerns the third and final stage. To ground the discussion, it helps to recall how the first two stages set the table for preference tuning.

### 1.1 Pre-training

The model is initialized from scratch and trained on massive text corpora (web pages, books, code) using next-token prediction. This step is extremely compute-heavy, requiring distributed training techniques such as ZeRO (stages 0 through 3) and model parallelism. At the end of pre-training, the model has deep knowledge of language structure and factual content, but all it can do is predict the next token. It is a capable autocompleter, not a helpful assistant.

### 1.2 Supervised Fine-Tuning (SFT)

The pre-trained model is adapted to a specific task -- typically conversational assistance -- by training it on a curated dataset of (prompt, desired response) pairs. This dataset is much smaller than the pre-training corpus but must be of much higher quality. Parameter-efficient methods like LoRA can be applied here, introducing low-rank matrices that are trained in place of the full weight matrices. After SFT, the model can follow instructions and produce structured answers, but it may still respond in ways that feel terse, unfriendly, or unsafe.

### 1.3 Preference Tuning (the focus of this lecture)

The SFT model is further aligned with human preferences. Rather than telling the model exactly which tokens to produce, preference tuning tells the model which style of output to favor and which to avoid. For example, given the prompt "Suggest a new activity I could do with my teddy bear," an SFT model might respond bluntly: "I would suggest you to not spend much time with your teddy bear at all." Preference tuning steers the model toward warmer, more helpful completions -- like suggesting specific activities and expressing enthusiasm -- without changing the factual content.

---

## 2. Why Preference Tuning is Needed Beyond SFT

A natural question is why a third training stage is necessary at all when SFT already teaches the model how to behave. There are four distinct reasons.

**Ease of data construction.** Writing a perfect response from scratch (as SFT requires) is much harder than comparing two existing responses and saying which is better. If someone asked you to write a great poem from scratch, that would be difficult. Choosing which of two poems is better is far easier. Preference data is cheaper and faster to collect.

**Distribution sensitivity of SFT data.** The SFT dataset must be carefully balanced across prompt types. If the model misbehaves on a specific kind of prompt, the naive fix of adding corrective examples to the SFT data risks biasing the entire model's behavior. Preference tuning provides a less disruptive way to address targeted misbehaviors without altering the overall prompt distribution.

**Negative signal injection.** SFT only provides positive supervision -- it tells the model what it should generate. It has no mechanism to tell the model what it should not generate. Preference tuning fills this gap by explicitly pairing preferred and dispreferred outputs, letting the model learn to downweight undesirable behaviors.

**Preserving SFT data quality.** Dumping every model failure into the SFT dataset degrades its quality. Preference tuning keeps the high-quality SFT data intact and addresses alignment issues through a separate mechanism. That said, preference tuning is not a cure-all: if the model is misbehaving broadly, the root cause may be a flawed SFT dataset, and fixing that dataset directly may be the better remedy.

One important clarification: LoRA (from the SFT stage) and preference tuning are not mutually exclusive. LoRA is a parameter-efficiency method that controls which weights get updated; preference tuning defines a different objective function. You can use LoRA during preference tuning just as you would during SFT.

---

## 3. Preference Data Collection

Before any preference-based training can begin, you need a dataset of preference pairs. The setup is: given a prompt, the model produces responses, and a rater indicates which response is better.

### 3.1 Three Rating Paradigms

**Pointwise scoring.** Each response receives an absolute score (e.g., 0.9 for a good poem, 0.2 for a bad one). This is cognitively difficult for human raters because it requires calibrating a numerical scale. How do you decide whether a poem is a 0.7 or a 0.8?

**Pairwise comparison.** Two responses are shown side-by-side and the rater simply picks which one is better. This is the dominant paradigm in practice because it is much easier for humans to make relative judgments than absolute ones.

**Listwise ranking.** The rater receives $n$ responses and ranks them from best to worst. This is more informative than pairwise comparison but harder and slower for raters to perform, especially as $n$ grows.

In practice, the field has converged on **pairwise comparison** as the standard, typically on a binary scale (response A is better or worse than response B). Some efforts use a more granular scale (much better, better, slightly better, slightly worse, worse, much worse), but binary judgments are simpler, less noisy, and more commonly used.

### 3.2 Generating Response Pairs

To produce two different responses for the same prompt, the standard technique is to sample the model twice with a positive temperature. Because temperature introduces randomness into token selection (as covered in Lecture 3), two forward passes on the same prompt will yield different completions.

The prompts themselves should follow the distribution of real user queries -- they can come from production logs or from a curated set of representative prompts. The goal is for the preference data to cover the kinds of inputs the model will actually encounter.

### 3.3 Rating Methods

**Human ratings** remain the gold standard for RLHF (the "H" stands for "human"). Raters compare pairs and indicate a preference.

**LLM-as-a-judge** uses a separate, typically stronger, language model to evaluate which response is better. This approach scales more easily but introduces its own biases.

**Rule-based metrics** like BLEU and ROUGE can also be used, though they are less common in modern preference tuning because they capture surface-level overlap rather than the nuanced qualities (helpfulness, safety, tone) that preference tuning targets.

An alternative to sampling two responses is to identify a bad response in production logs and manually rewrite it into a good response. This produces high-quality pairs but is labor-intensive since it requires human writing, not just human judgment.

### 3.4 Challenges with Human Ratings

Many tasks involve subjective judgments where reasonable raters disagree. Clear, objective annotation guidelines are critical to reducing noise in the preference labels. Despite best efforts, some noise is unavoidable, and this imperfection in the training signal has downstream consequences (see reward hacking in Section 6.3).

---

## 4. RLHF Overview and RL Basics for LLMs

**RLHF** stands for Reinforcement Learning from Human Feedback. When the preference labels come from human raters, we call the method RLHF. When they come from an AI system (such as an LLM judge), the method is sometimes called RLAIF (Reinforcement Learning from AI Feedback).

RLHF has two stages:
1. **Train a reward model** that can score any (prompt, response) pair, distinguishing good outputs from bad ones.
2. **Use the reward model to align the LLM** through reinforcement learning, generating completions and updating the model's weights based on the rewards those completions receive.

### 4.1 Mapping RL Concepts onto Language Models

Reinforcement learning operates with a standard vocabulary of agents, states, actions, policies, and rewards. Each maps cleanly onto the LLM setting:

| RL Concept | LLM Equivalent |
|---|---|
| **Agent** | The LLM itself |
| **State** $s_t$ | The input so far (prompt + tokens generated up to step $t$) |
| **Action** $a_t$ | Predicting the next token |
| **Action space** | The token vocabulary |
| **Policy** $\pi_\theta(a_t \mid s_t)$ | The probability distribution over next tokens, as output by the LLM's forward pass |
| **Reward** | A score assigned to the full completion, from the reward model |

The policy $\pi_\theta(a_t \mid s_t)$ gives the probability of choosing action $a_t$ (i.e., a specific next token) given the current state $s_t$ (the prompt plus all tokens generated so far). The parameters $\theta$ are the LLM's weights, and the goal of training is to find $\theta$ such that the policy aligns with human preferences.

### 4.2 Sparse Signal

A key difference from SFT: during supervised fine-tuning, every token position provides a training signal (the model learns to predict each next token). In the RL setting, the reward is assigned to the entire completion as a whole -- roughly one signal per generation rather than one signal per token. This makes RLHF a much sparser learning signal, which is why RLHF is generally viewed as a refinement step rather than a replacement for SFT.

---

## 5. Reward Model Training

The first stage of RLHF is to train a model that can evaluate (prompt, response) pairs and produce a scalar score indicating quality. This reward model is trained on the preference data collected in Section 3.

### 5.1 The Bradley-Terry Formulation

The reward model is trained using the **Bradley-Terry model**, a classical statistical framework for pairwise comparisons. It defines the probability that response $y_i$ is preferred over response $y_j$ as:

$$P(y_i \succ y_j) = \frac{e^{R_i}}{e^{R_i} + e^{R_j}} = \sigma(R_i - R_j)$$

where:
- $R_i = R(x, y_i)$ is the reward model's score for response $y_i$ given prompt $x$
- $R_j = R(x, y_j)$ is the reward model's score for response $y_j$ given prompt $x$
- $\sigma$ is the sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$

The sigmoid maps the difference in scores to a probability between 0 and 1. When $R_i \gg R_j$, the sigmoid output approaches 1 (high confidence that $y_i$ is better). When $R_i \ll R_j$, it approaches 0. When $R_i = R_j$, it equals 0.5 (a coin flip).

### 5.2 Deriving the Loss Function

Given a preference dataset of $n$ pairs $\{(y_w^{(i)}, y_l^{(i)})\}_{i=1}^{n}$, where $y_w$ is the winning (preferred) response and $y_l$ is the losing (dispreferred) response, we want to find model parameters $\theta$ that maximize the likelihood of observing these preferences.

Assuming pairs are independent, the likelihood is:

$$\mathcal{L}(\theta) = \prod_{i=1}^{n} P(y_w^{(i)} \succ y_l^{(i)}) = \prod_{i=1}^{n} \sigma\!\left(R_\theta(x^{(i)}, y_w^{(i)}) - R_\theta(x^{(i)}, y_l^{(i)})\right)$$

Taking the log converts the product to a sum (for numerical stability), and negating it turns maximization into minimization:

$$\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}\left[\log \sigma\!\left(R_\theta(x, y_w) - R_\theta(x, y_l)\right)\right]$$

This is the reward model training loss. It pushes the model to assign higher scores to preferred responses and lower scores to dispreferred ones.

### 5.3 Pairwise Training, Pointwise Inference

An elegant property of this setup: the loss function is pairwise (it requires two responses to compute a gradient), but the trained reward model is pointwise at inference time. It takes a single (prompt, response) pair and outputs a single scalar score. The pairwise training teaches it what "good" and "bad" look like, but once trained, it can evaluate any response independently. The reward model might output scores like 0.8 for a helpful response and -2.0 for an unhelpful one -- continuous values on an unconstrained scale.

### 5.4 Practical Details

**Data scale.** Reward model training typically requires tens of thousands of preference pairs or more.

**Model architecture.** The reward model is commonly a decoder-only LLM with a classification (scalar regression) head appended after the final token position. An encoder-only model like BERT, projecting the CLS token embedding to a scalar, is also viable but less common today since the field has converged on decoder-only architectures.

**Reward dimensions.** The preference labels should correspond to a defined dimension of quality: helpfulness, safety, friendliness, factual accuracy, etc. Different dimensions may require separate reward models. A single "holistic" score is also possible but less interpretable.

**Score normalization.** The raw reward values have no fixed scale. In practice, scores are normalized (e.g., standardized across a batch) before being used in the RL training loop. For the best-of-$n$ sampling method (Section 7), the scale does not matter because only the relative ordering matters.

**Evaluation.** RewardBench is a popular benchmark for evaluating reward model quality.

---

## 6. Policy Optimization with PPO

The second stage of RLHF uses the trained (and now frozen) reward model to update the LLM's weights so that it generates higher-reward completions. The standard algorithm for this is **Proximal Policy Optimization (PPO)**.

### 6.1 The Core Training Loop

1. Sample a prompt $x$ from the training distribution.
2. The LLM (the policy being trained) generates a full completion $\hat{y}$ -- also called a **rollout**.
3. The frozen reward model scores the (prompt, completion) pair, producing a reward $R(x, \hat{y})$.
4. The reward signal is used to update the LLM's weights so that it is more likely to produce high-reward completions and less likely to produce low-reward ones.

This is **on-policy training**: at each iteration, the model generates outputs from its current policy and optimizes based on those outputs. This contrasts with SFT, which is **off-policy** -- training on a fixed dataset of responses that were not generated by the model being trained.

### 6.2 Why Constrain Updates: Three Reasons

The PPO loss function has two components: maximize rewards, and stay close to the original SFT model (the **reference model**). The word "proximal" in PPO refers to this constraint. Three reasons justify it:

**Catastrophic forgetting.** The pre-trained and fine-tuned model contains vast knowledge about language, code, and the world. Aggressive updates risk destroying this knowledge in pursuit of higher rewards.

**Reward hacking.** The reward model is an imperfect proxy for what humans actually want. If the LLM optimizes too aggressively against this proxy, it will find exploits -- outputs that score highly with the reward model but are not genuinely good. Consider an analogy: if a lecturer's reward is the volume of applause at the end, the lecturer might optimize for jokes instead of informative content. The applause metric is maximized, but the true objective (informative lecture) is not. In LLM terms, the model might learn stylistic tricks that fool the reward model without improving actual helpfulness.

**Training instability.** Large, unconstrained policy updates can cause the training process to diverge or oscillate, making it difficult to converge to a good solution.

### 6.3 KL Divergence as a Proximity Measure

To measure how far the current policy has drifted from the reference model, PPO uses the **KL divergence** (Kullback-Leibler divergence):

$$D_{\text{KL}}(P \| Q) = \sum_i P_i \log \frac{P_i}{Q_i}$$

where $P$ and $Q$ are two probability distributions. The KL divergence has two important properties:
- It is always non-negative: $D_{\text{KL}}(P \| Q) \geq 0$, provable via Jensen's inequality.
- It equals zero if and only if $P = Q$.

It is not a true distance metric (it is asymmetric: $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$), but it serves well as a measure of distributional divergence.

In the PPO loss, minimizing the KL divergence between the current policy $\pi_\theta$ and the reference policy $\pi_{\text{ref}}$ prevents the model from straying too far from the SFT baseline.

### 6.4 Advantage Estimation and the Value Function

PPO does not optimize raw rewards directly. Instead, it optimizes the **advantage** $A$, which measures how much better a particular output is compared to what you would expect on average:

$$A = R - V$$

where $V$ is the **value function** -- an estimate of the expected reward if the model continues generating according to its current policy from a given partial state.

The value function $V(s_t)$ takes a partial input (prompt plus tokens generated so far, up to step $t$) and predicts the final reward that would result from continuing generation under the current policy. Unlike the reward model (which scores complete outputs), the value function provides token-level estimates.

**Why use advantages instead of raw rewards?** Subtracting the baseline $V$ reduces variance in the gradient estimates, making training faster and more stable. A raw reward of 0.8 is ambiguous -- is that good or bad? But an advantage of +0.3 (meaning 0.3 better than expected) is unambiguously positive signal.

The value function is trained jointly with the policy, typically as a regression head on the LLM that predicts a scalar value at each token position. The method for computing advantages from value function estimates is called **Generalized Advantage Estimation (GAE)**, described in the paper "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al.). GAE introduces its own hyperparameters that control the bias-variance tradeoff in the advantage estimates.

### 6.5 PPO-Clip

The first variant of the PPO loss prevents excessively large updates by **clipping** the probability ratio between the current and previous policy:

$$\mathcal{L}^{\text{CLIP}} = \min\!\left(r(\theta) \cdot A, \;\text{clip}\!\left(r(\theta), 1-\epsilon, 1+\epsilon\right) \cdot A\right)$$

where:
- $r(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the **probability ratio** between the current policy and the policy from the previous iteration (not the reference SFT model, but the model at the prior training step)
- $A$ is the advantage estimate
- $\epsilon$ is a hyperparameter controlling how far $r(\theta)$ can move from 1.0

**Important notation clarification:** Despite using $r(\theta)$, this is not the reward model. It is a ratio of policy probabilities. Also, despite being written as "$\mathcal{L} =$", this expression is **maximized**, not minimized.

**When the advantage is positive** (the action was better than expected), the loss increases as $r(\theta)$ increases -- reinforcing the action by making it more likely. But the clipping at $1 + \epsilon$ caps the reinforcement, preventing too large an update in one step.

**When the advantage is negative** (the action was worse than expected), the loss increases as $r(\theta)$ decreases -- making the action less likely. But the clipping at $1 - \epsilon$ caps this suppression, again preventing too large an update.

The clipping is analogous in spirit to a ReLU cutoff: the function is piecewise linear, and standard automatic differentiation handles the non-differentiable kink at the clip boundary the same way it handles ReLU.

The key distinction: $\pi_{\theta_{\text{old}}}$ in the ratio is the policy from the **previous RL iteration**, not the SFT reference model. This constrains step-to-step update magnitude for training stability.

### 6.6 PPO-KL

The second variant replaces clipping with an explicit KL penalty:

$$\mathcal{L}^{\text{KL}} = r(\theta) \cdot A - \beta \cdot D_{\text{KL}}\!\left(\pi_{\theta_{\text{old}}} \| \pi_\theta\right)$$

where $\beta$ controls the strength of the KL penalty. In the original PPO paper (2017), $\pi_{\theta_{\text{old}}}$ referred to the previous iteration. In modern LLM applications, practitioners often substitute the SFT reference model $\pi_{\text{ref}}$ for the KL term, and some implementations combine both variants: using clipping with respect to the previous iteration (for step-level stability) and a KL penalty with respect to the reference model (to prevent drift from the SFT baseline).

### 6.7 Models Required for PPO

PPO-based RLHF requires four models to be loaded during training:

| Model | Role | Trainable? |
|---|---|---|
| Policy $\pi_\theta$ | The LLM being aligned | Yes |
| Value function $V$ | Estimates expected rewards for advantage computation | Yes (trained jointly) |
| Reward model $R$ | Scores completions | No (frozen) |
| Reference model $\pi_{\text{ref}}$ | The SFT model, for KL divergence computation | No (frozen) |

This is a heavy computational footprint -- four full model copies (or near-copies) must reside in memory simultaneously. The training data scale is typically at least 100k prompts, larger than the reward model training set.

---

## 7. Best-of-N Sampling

For practitioners who have a trained reward model but want to avoid the complexity of RL training entirely, **best-of-$n$ sampling** (also called **rejection sampling**) offers a simple alternative.

### 7.1 How It Works

1. Given a prompt, generate $n$ completions from the SFT model (using positive temperature for diversity).
2. Score all $n$ completions with the reward model.
3. Return the highest-scoring completion to the user.

For example, given "Suggest a new activity I could do with my teddy bear," the model generates three completions:
- "Of course! Teddy bears make awesome companions..." (score: 0.8)
- "I would suggest you not spend time with your teddy bear." (score: -2.0)
- "Take your teddy bear to a picnic!" (score: 0.4)

Best-of-$n$ returns the first completion (score 0.8).

### 7.2 Trade-offs

**No training required.** The SFT model is used as-is; no RL training loop, no value function, no training instabilities to manage.

**Inference cost scales linearly.** Generating $n$ completions costs roughly $n$ times as much as generating one. For high-traffic applications, this multiplication can be prohibitive. Even with infinite compute and full parallelism, the latency is determined by the slowest of the $n$ generations, and the distribution of the maximum of $n$ latency samples shifts rightward -- so wall-clock time increases even in the best case.

**Bounded by model quality.** If the SFT model is poor, all $n$ completions may be poor, and picking the best of a bad set still yields a bad result. The method requires the model to already be reasonably capable.

**Scale-invariant.** Because only the ranking matters (not the absolute score), the method is robust to reward model score calibration issues. Any monotonic transformation of the scores produces the same selection.

---

## 8. Direct Preference Optimization (DPO)

DPO represents a fundamentally different approach to preference tuning. Instead of the two-stage RLHF pipeline (train a reward model, then do RL), DPO optimizes a single supervised loss function that directly aligns the model with preference data. The paper is titled "Your Language Model Is Secretly a Reward Model" -- a name that foreshadows its key insight.

### 8.1 Motivation

The complaints against RLHF motivate the search for alternatives:
- Four models must be maintained in memory during PPO training.
- The two-stage process creates hard dependencies: if the reward model has problems, you must retrain it and then redo the RL stage.
- PPO introduces many hyperparameters ($\beta$, $\epsilon$, GAE parameters) and is notoriously difficult to tune.
- Training instability requires careful monitoring with imperfect metrics (average reward is the standard monitor, but it is not as informative as the cross-entropy loss used in SFT and pre-training).
- On-policy training requires the model to explore diverse completions, adding another source of complexity.
- RL expertise is not widespread, and it is not obvious that RL is strictly necessary for this task.

### 8.2 Derivation

DPO's derivation starts from the same PPO objective and arrives at a closed-form solution that eliminates the reward model entirely.

**Step 1: Write the PPO objective.** The goal is to maximize rewards while penalizing divergence from the reference policy:

$$\max_{\pi_\theta} \; \mathbb{E}\!\left[R(x, y)\right] - \beta \cdot D_{\text{KL}}\!\left(\pi_\theta \| \pi_{\text{ref}}\right)$$

**Step 2: Solve for the optimal policy.** The paper derives the closed-form optimal policy $\pi^*$ as a function of the reward $R$:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \cdot \exp\!\left(\frac{1}{\beta} R(x, y)\right)$$

where $Z(x)$ is a partition function (normalization constant) that depends only on $x$ and the reference policy. This step involves no additional assumptions -- it is a direct analytical solution to the optimization problem.

**Step 3: Rearrange to express $R$ in terms of $\pi$.** Solving the above for $R$:

$$R(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

This is the paper's central insight: the reward can be written as a function of the policy itself. The language model "is secretly a reward model" because the log-ratio of policy probabilities implicitly encodes what a reward model would have learned.

**Step 4: Substitute into the Bradley-Terry formulation.** Plugging this expression for $R$ into the preference probability $P(y_w \succ y_l) = \sigma(R_w - R_l)$, the $Z(x)$ terms cancel:

$$P(y_w \succ y_l) = \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

**Step 5: Form the loss function.** Taking the negative log-likelihood:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

where:
- $\pi_\theta$ is the policy being trained
- $\pi_{\text{ref}}$ is the frozen SFT reference model
- $y_w$ is the winning (preferred) completion
- $y_l$ is the losing (dispreferred) completion
- $\beta$ is the same trade-off hyperparameter from the PPO objective (typically around 0.1), controlling how strongly to penalize divergence from the reference model
- $\sigma$ is the sigmoid function

### 8.3 What the Loss Function Does

Each term $\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ plays the role of an implicit reward. The loss pushes the model to increase the probability of preferred completions relative to the reference model, and decrease the probability of dispreferred completions relative to the reference model, with the constraint that neither shift should be too extreme (controlled by $\beta$).

### 8.4 Practical Advantages

**Only two models needed.** DPO requires only the policy $\pi_\theta$ (trainable) and the reference model $\pi_{\text{ref}}$ (frozen). No reward model, no value function.

**Supervised training.** The loss function is computed directly from the preference pairs using standard forward passes and backpropagation. No rollouts, no reward scoring, no advantage estimation.

**Fewer hyperparameters.** The primary hyperparameter is $\beta$. There is no $\epsilon$ for clipping, no GAE parameters, no reward model training to tune separately.

**No RL expertise required.** DPO reduces preference tuning to a standard supervised learning problem, making it accessible to practitioners without reinforcement learning backgrounds.

### 8.5 DPO vs. RLHF: Trade-offs

| Dimension | RLHF (PPO) | DPO |
|---|---|---|
| Training stages | Two (reward model + RL) | One (direct optimization) |
| Models in memory | Four | Two |
| Training complexity | High (many hyperparameters, instabilities) | Low (supervised loss) |
| RL expertise needed | Yes | No |
| Peak performance | Generally higher | Slightly lower |
| Distribution shift risk | Lower (on-policy generation) | Higher (off-policy data) |

**The distribution shift problem.** DPO's main weakness is that it trains on a fixed preference dataset -- the completions in the dataset were generated by some model (possibly an earlier version of the policy or a different model), not by the policy currently being trained. This is an off-policy setup, and the mismatch between the training data distribution and the model's actual output distribution can degrade performance. RLHF avoids this because it generates fresh completions from the current policy at each training step (on-policy).

Mitigations include running SFT on the preference data before DPO, or generating preference pairs using the model's own outputs, but these add cost and complexity that partially offset DPO's simplicity advantages.

**The practical calculus.** If you want the highest possible alignment quality and have the compute budget and RL expertise to manage PPO training, RLHF tends to produce better results. If you want a quick, effective preference tuning step with minimal infrastructure complexity and are willing to accept marginally lower ceiling performance, DPO is the pragmatic choice. The referenced study in the lecture confirms that PPO generally outperforms DPO on benchmarks, but the gap may not justify the additional engineering effort for many use cases.

---

## 9. Illustrative Example: Preference Tuning in Action

To make the three-stage pipeline concrete, consider the running example used throughout the course. The prompt is: "Can I put my teddy bear in the washer?"

**After SFT**, the model responds: "No, it might get damaged. Try hand washing it instead." This is factually correct -- teddy bears should indeed be hand-washed -- but the tone is blunt. For someone who loves their teddy bear, it feels dismissive.

**After preference tuning**, the model responds: "It's better not to -- your teddy bear could get hurt. A gentle hand wash is safer." The same facts are conveyed, but the tone is warmer and more considerate.

Preference tuning did not teach the model new facts. It adjusted the distribution of completions so that the model favors gentler, more empathetic phrasing when the content allows for it. This is precisely the kind of alignment that is difficult to achieve through SFT alone without distorting the training data distribution.

---

## Key Takeaways

1. **Preference tuning** is the third stage of the LLM training pipeline (after pre-training and SFT). It aligns the model with human preferences about tone, helpfulness, and safety, using preference pairs rather than gold-standard responses. It also injects negative signal, which SFT cannot provide.

2. **Pairwise comparison** is the standard format for preference data because it is far easier for humans to say which of two responses is better than to assign absolute scores. Data is generated by sampling the model twice with positive temperature and having raters (human or AI) pick the winner.

3. **The reward model** is trained using the Bradley-Terry formulation, with loss $\mathcal{L} = -\mathbb{E}[\log \sigma(R_w - R_l)]$. It is trained in a pairwise manner but operates pointwise at inference, assigning a scalar score to any single (prompt, response) pair.

4. **PPO** aligns the LLM by maximizing advantages (reward relative to baseline) while constraining the policy to stay close to the reference model, using either clipping (PPO-Clip) or a KL penalty (PPO-KL). It requires four models in memory: the policy, value function, reward model, and reference model.

5. **Reward hacking** is a central risk: the reward model is an imperfect proxy for human preferences, and over-optimizing against it produces outputs that game the metric without genuinely improving quality. The KL constraint and clipping mechanisms exist specifically to mitigate this.

6. **Best-of-$n$ sampling** sidesteps RL entirely by generating multiple completions and selecting the highest-scoring one via the reward model. It trades training cost for inference cost and works well when serving traffic is low.

7. **DPO** eliminates the reward model entirely by deriving a closed-form supervised loss from the same objective that PPO optimizes. It requires only two models (policy and reference), has fewer hyperparameters, and needs no RL expertise, but is susceptible to distribution shift because it trains on fixed off-policy data. PPO generally achieves higher peak performance, but DPO offers a far simpler path to effective preference tuning.
