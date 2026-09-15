# Lecture 8: LLM Evaluation - Stanford CME 295

**Date:** November 21, 2025
**Instructors:** Afshine Amidi & Shervine Amidi (Adjunct Lecturers, Stanford)
**Course:** CME 295 - Transformers and Large Language Models, Autumn 2025

---

## 1. Defining Evaluation

When we say "evaluate an LLM," the phrase can mean many different things. It might refer to output quality (coherence, factuality, relevance), system-level metrics (latency, uptime, cost), or alignment properties (safety, tone, format compliance). This lecture focuses specifically on **output quality** -- quantifying how good the actual response is.

This is a fundamentally hard problem. LLMs are text-to-text models that produce free-form output spanning natural language, code, mathematical reasoning, and more. There is no single universal metric that captures quality across all these domains. The lecture walks through the progression of evaluation techniques -- from human ratings to rule-based metrics to LLM-as-a-Judge -- examining the strengths and failure modes of each.

The motivation is practical: if you cannot measure the performance of your LLM, you do not know what to improve. Evaluation is the feedback loop that drives iteration.

---

## 2. Human Evaluation

### 2.1 The Ideal Scenario and Its Limitations

The ideal evaluation pipeline would work as follows: give a prompt to the LLM, collect the response, ask a human to rate it, and repeat. After accumulating enough ratings, aggregate them to quantify overall model performance.

This approach has two fundamental problems. First, it is expensive and slow -- rating thousands of LLM outputs by hand takes significant time and budget. Second, human judgment itself can be subjective, which introduces noise into the ratings.

### 2.2 Subjectivity and Inter-Rater Agreement

Consider a prompt like "What birthday gift should I get?" and a response of "A teddy bear is almost always a sweet gift. Just pick one that feels right for you." One human rater might judge this as useful because it gives a concrete suggestion. Another might judge it as not useful because it lacks specificity -- which teddy bear? What size? What price range?

This subjectivity means that even with human evaluators, there is no guaranteed consensus. The concern is captured by the concept of **inter-rater agreement**: how consistently do different raters assign the same scores?

### 2.3 Why Raw Agreement Rate Is Misleading

A natural first metric is the **agreement rate** -- the proportion of times two raters give the same label. But this metric is misleading because it does not account for agreement that would occur by pure chance.

Consider two raters, Alice and Bob, assigning binary labels (good/not good) independently at random. Alice labels "good" with probability $p_A$, and Bob with probability $p_B$. The expected agreement rate by chance is:

$$P(\text{agree by chance}) = p_A \cdot p_B + (1 - p_A)(1 - p_B)$$

where the first term is the probability both say "good" and the second is the probability both say "not good."

If $p_A = p_B = 0.5$, the agreement rate by chance is $0.5^2 + 0.5^2 = 0.5$, meaning 50% agreement from pure randomness. If both raters are heavily biased toward one label (e.g., $p_A = p_B = 0.9$), the chance agreement is even higher: $0.81 + 0.01 = 0.82$. So a raw agreement rate of 82% might look good but could be entirely explained by chance.

### 2.4 Cohen's Kappa and Extensions

To account for chance agreement, practitioners use metrics that normalize the observed agreement against the random baseline. **Cohen's kappa** is the standard choice for two raters:

$$\kappa = \frac{p_{\text{observed}} - p_{\text{chance}}}{1 - p_{\text{chance}}}$$

where $p_{\text{observed}}$ is the actual agreement rate and $p_{\text{chance}}$ is the expected agreement rate under random labeling.

- If $p_{\text{observed}} = 1$ (perfect agreement), then $\kappa = 1$.
- If $p_{\text{observed}} = p_{\text{chance}}$ (no better than random), then $\kappa = 0$.
- If $p_{\text{observed}} < p_{\text{chance}}$ (worse than random), then $\kappa < 0$.

For settings with more than two raters, extensions exist: **Fleiss's kappa** generalizes Cohen's kappa to multiple raters, and **Krippendorff's alpha** handles multiple raters with potentially missing data. All share the same underlying principle: measure how much better actual agreement is compared to the random baseline.

In practice, teams track these agreement metrics as a health indicator. When agreement drops below an acceptable threshold, raters hold alignment sessions to clarify the rating guidelines and re-calibrate. The goal is not to force agreement but to ensure disagreements reflect genuine ambiguity rather than unclear instructions.

---

## 3. Rule-Based Metrics

### 3.1 The Reference-Based Approach

Since rating every output by hand is impractical, one alternative is to collect human-written reference answers for a fixed set of prompts once, and then use automated metrics to compare LLM outputs against those references. This allows iterating on the model without re-engaging human raters each time -- a significant cost reduction.

The challenge is designing metrics that are flexible enough to handle the reality of natural language, where the same idea can be expressed in many different ways.

### 3.2 METEOR

**METEOR** (Metric for Evaluation of Translation with Explicit Ordering) was designed for machine translation evaluation. It computes a weighted F-score between the predicted and reference outputs, then applies a penalty for incorrect word ordering.

$$\text{METEOR} = F_{\text{score}} \times (1 - \text{Penalty})$$

The F-score component is a parameterized harmonic mean of precision and recall:

$$F_{\text{score}} = \frac{P \cdot R}{\alpha \cdot P + (1 - \alpha) \cdot R}$$

where:
- $P$ (precision) is the proportion of unigrams in the prediction that match unigrams in the reference.
- $R$ (recall) is the proportion of unigrams in the reference that match unigrams in the prediction.
- $\alpha$ is a hyperparameter controlling the precision-recall trade-off.

The penalty term incentivizes correct ordering:

$$\text{Penalty} = \gamma \left(\frac{C}{M}\right)^\beta$$

where:
- $C$ is the number of contiguous matched chunks (groups of consecutive matching unigrams).
- $M$ is the total number of matched unigrams.
- $\gamma$ and $\beta$ are hyperparameters.

A low $C/M$ ratio means the matched unigrams form long contiguous stretches, indicating that the word order in the prediction closely follows the reference. A high ratio means the matches are fragmented, indicating disordered output. Higher METEOR scores indicate better translations.

METEOR expands the definition of "matched unigrams" beyond exact string matches to include synonyms and words sharing the same stem, which adds some flexibility. However, the metric is still fundamentally brittle -- it relies on surface-level token overlap.

### 3.3 BLEU and ROUGE

**BLEU** (Bilingual Evaluation Understudy) is a precision-focused metric that counts matching n-grams between prediction and reference. It includes a **brevity penalty** to discourage gaming the metric by producing very short outputs (which would have high precision by cherry-picking only safe n-grams).

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) is typically used for summarization and emphasizes recall -- how many n-grams from the reference appear in the prediction. It has several variants (ROUGE-1 for unigrams, ROUGE-2 for bigrams, ROUGE-L for longest common subsequence).

### 3.4 Fundamental Limitations of Rule-Based Metrics

All three metrics share critical weaknesses:

**No tolerance for stylistic variation.** Consider three ways of expressing the same idea:
- "A plush teddy bear can comfort a child during bedtime."
- "Soft stuffed bears often help kids feel safe as they fall asleep."
- "Many youngsters rest more easily at night when they cuddle a gentle toy companion."

These sentences are semantically equivalent, but they share almost no n-grams. Any metric based on token overlap would assign low scores to valid paraphrases, penalizing stylistic diversity rather than actual quality failures.

**Weak correlation with human judgment.** Despite the hyperparameters that were tuned to maximize correlation with human ratings, these metrics remain only loosely correlated with what humans actually consider good output.

**Still require human references.** The whole point was to reduce reliance on humans, but these metrics still need human-written reference answers to function at all.

These limitations motivate the move to LLM-as-a-Judge.

---

## 4. LLM-as-a-Judge

### 4.1 Core Idea

The central insight is that LLMs, having been pretrained on vast corpora and aligned to human preferences through techniques like RLHF, already encode substantial knowledge about what constitutes a good response. Instead of comparing outputs against references with brittle token-overlap metrics, we can use another LLM to evaluate the response directly.

**LLM-as-a-Judge** (a term introduced in a 2023 paper) takes three inputs:
1. The **prompt** used to generate the response.
2. The **response** being evaluated.
3. The **criteria** along which to evaluate (e.g., usefulness, factuality, relevance).

It produces two outputs:
1. A **score** (typically binary: pass/fail).
2. A **rationale** explaining why that score was assigned.

The rationale is the key differentiator from prior methods. Rule-based metrics produce opaque numbers -- a BLEU score of 0.34 does not tell you what went wrong. An LLM judge can explain that the response lacked specificity, contradicted the prompt, or contained unsupported claims.

### 4.2 Prompt Structure and Rationale-Before-Score

A typical judge prompt follows this structure: state the evaluation criteria, provide the original prompt and the model response, then ask the judge to return a rationale followed by a score.

The ordering matters -- asking for the **rationale before the score** empirically improves judgment quality. This mirrors the chain-of-thought reasoning approach seen in reasoning models (Lecture 6): by verbalizing its analysis first, the judge effectively "thinks through" the evaluation before committing to a verdict. If the score came first, the rationale might simply post-hoc rationalize the initial snap judgment.

### 4.3 Structured Output for Reliable Parsing

A raw LLM-as-a-Judge call is not guaranteed to produce output in a parseable format. The model might embed the score in a sentence, omit it entirely, or format the rationale inconsistently. Since the judge's output must be programmatically consumed, reliability matters.

The solution is **structured output** (also called constrained or guided decoding), which was covered in Lecture 3 (slide 65). By defining a schema -- for example, a class with `rationale: str` and `score: int` fields -- and passing it to the API's structured output parameter, the decoding process is constrained to only emit tokens that produce valid output conforming to that schema. Major providers (OpenAI, Anthropic, Google) all support this capability.

### 4.4 Two Flavors of LLM-as-a-Judge

**Pointwise evaluation:** A single response is evaluated in isolation. The judge assigns a score (pass/fail or on a rating scale) based on the criteria. This is the simpler setup and is used when you need to evaluate one model's outputs independently.

**Pairwise evaluation:** Two responses are presented side by side, and the judge decides which is better. This setup is particularly useful for generating synthetic preference data for reward model training (as discussed in Lecture 5 on preference tuning). Instead of collecting expensive human preference labels, you can use an LLM judge to compare response pairs and produce the preference signal needed for DPO or RLHF.

### 4.5 Key Benefits Over Prior Methods

First, LLM-as-a-Judge requires **no reference text** and no human ratings to get started. The judge draws on knowledge acquired during pretraining and alignment. Second, the score is **interpretable** -- the rationale explains the judgment, making it actionable for debugging and improvement.

---

## 5. Biases in LLM-as-a-Judge

LLM judges are not perfect. They exhibit systematic biases that can distort evaluation results if left unaddressed.

### 5.1 Position Bias

In pairwise evaluation, the judge may favor whichever response appears first in the prompt, regardless of actual quality. If you ask "Is Response A or Response B better?", the judge may lean toward A simply because it was presented first.

**Mitigation:** Run the comparison twice with the order swapped -- once as (A, B) and once as (B, A). If both orderings produce the same winner, the result is reliable. If the winner changes, the judgment is unreliable and should be handled differently (e.g., marked as a tie or escalated to human review). More advanced approaches involve modifying position embeddings, but the swap-and-vote method is the standard practical remedy.

### 5.2 Verbosity Bias

Judges tend to prefer longer, more detailed responses over shorter, concise ones -- even when the shorter response is more accurate or appropriate. The bias is toward verbosity for its own sake, not because the additional content adds value.

**Mitigation strategies:**
- **Explicit instructions:** Add guidance in the judge prompt stating that length should not influence the rating and that concise, accurate responses are equally valid.
- **In-context examples:** Provide few-shot examples where a short response is rated higher than a verbose one, demonstrating that brevity is acceptable.
- **Length penalty:** In a pointwise setup, rate each response independently and then apply a penalty proportional to output length, discounting inflated scores from verbose responses.

### 5.3 Self-Enhancement Bias

When a model judges output generated by itself (or a closely related model), it tends to assign higher scores to its own outputs. The intuition is probabilistic: if the model generated a particular sequence, it assigned high likelihood to that sequence during generation, and the same priors carry into evaluation.

**Mitigation:** Use a different model for judging than the one used for generation. In practice, this constraint is imperfect -- many frontier models are trained on similar data mixtures and may share biases. But using a distinct model, ideally one with greater capacity and stronger reasoning abilities, reduces the risk. The general guidance is: the judge should be a **larger, more capable model** than the one being evaluated, so it can identify subtle quality differences rather than defaulting to self-familiar patterns.

### 5.4 Other Potential Biases

These three are not exhaustive. Another common bias is **misalignment with ground truth** -- the judge may systematically favor one label or style that diverges from what human raters would prefer. The full taxonomy of judge biases is an active area of research.

---

## 6. Best Practices for LLM-as-a-Judge

The lecture consolidates several practical guidelines for running LLM-as-a-Judge effectively:

1. **Use crisp, explicit guidelines.** Vague criteria like "Is this response good?" leave too much room for interpretation. Instead, specify exactly what constitutes a pass or fail: "Does the response directly answer the user's question with at least one specific, actionable recommendation?"

2. **Prefer binary scales.** A pass/fail score is easier for both the judge and human calibrators to work with. Granular scales (1-5 or 1-10) introduce noise without proportional signal gain. The judgment task is simpler, the agreement between LLM and human raters is higher, and downstream analysis is cleaner.

3. **Output rationale before score.** As discussed in Section 4.2, this mirrors chain-of-thought reasoning and improves judgment quality.

4. **Actively mitigate known biases.** Apply the position-swapping, anti-verbosity instructions, and cross-model judging techniques described in Section 5.

5. **Calibrate against human ratings.** While LLM-as-a-Judge does not require human ratings to function, periodically comparing judge scores against human ratings is essential. Run correlation analysis between the two, and use the results to refine the judge prompt. The risk of skipping this step is optimizing against a proxy (the judge's score) that drifts from the true target (human satisfaction).

6. **Use low temperature.** For evaluation tasks, set the temperature to 0.1 or 0.2. The goal is reproducibility -- running the same evaluation twice should produce the same results. High temperature introduces unnecessary variance.

---

## 7. Evaluation Dimensions

LLM output quality can be assessed along two broad axes:

**Task performance:** Was the response useful? Was it factually correct? Was it relevant to the prompt? Did it complete the task the user asked for?

**Format and alignment:** Was the tone appropriate? Did the response follow the requested style or format? Were there any unsafe elements in the response?

Both axes matter, but factuality deserves special treatment because it requires a more sophisticated evaluation pipeline than a single judge call.

---

## 8. Factuality Evaluation

### 8.1 The Challenge of Nuance

Consider the following LLM output: "Teddy bears, first created in the 1920s, were named after President Theodore Roosevelt after he proudly wanted to shoot a captured bear on a hunting trip."

This text contains two factual errors: teddy bears were first created in the early 1900s (not the 1920s), and Roosevelt refused to shoot the bear (he did not "proudly want to"). But the text also contains correct facts -- it correctly identifies the connection to Theodore Roosevelt and to a hunting trip.

A binary pass/fail on the entire text would be too coarse. We need to capture the degree of factual accuracy, recognizing that some parts are correct while others are not.

### 8.2 Fact Decomposition and Verification

The standard approach proceeds in stages:

**Step 1 -- Decompose into atomic facts.** Use an LLM call to break the original text into individual factual claims. For the example above, this yields:
1. Teddy bears were first created in the 1920s.
2. Teddy bears were named after President Theodore Roosevelt.
3. Roosevelt proudly wanted to shoot a captured bear.
4. The event occurred on a hunting trip.

**Step 2 -- Verify each fact independently.** For each atomic fact, determine whether it is correct or incorrect using a binary judgment. The verification process typically involves external tools -- RAG against a knowledge base, web search, or other retrieval methods. Each fact check may itself be an LLM call grounded by retrieved evidence.

**Step 3 -- Aggregate with optional weighting.** Combine the per-fact verdicts into an overall factuality score:

$$\text{Factuality} = \frac{\sum_{i=1}^{n} \alpha_i \cdot \mathbb{1}[\text{fact}_i \text{ is correct}]}{\sum_{i=1}^{n} \alpha_i}$$

where:
- $n$ is the total number of decomposed facts.
- $\alpha_i$ is the importance weight for fact $i$.
- $\mathbb{1}[\cdot]$ is the indicator function (1 if the fact is correct, 0 otherwise).

The weights $\alpha_i$ can be uniform (all facts equally important) or can reflect the relative significance of each fact. For instance, correctly attributing the teddy bear's name to Roosevelt might be weighted more heavily than the exact decade of creation.

In the running example: facts 2 and 4 are correct, facts 1 and 3 are incorrect. With equal weighting, the factuality score is $2/4 = 0.5$. With weights reflecting importance (say, $\alpha = [0.5, 1.0, 0.5, 0.5]$), the score would be $(1.0 + 0.5) / (0.5 + 1.0 + 0.5 + 0.5) = 0.6$.

---

## 9. Agent Evaluation

### 9.1 The Agent Loop Revisited

As covered in Lecture 7, agentic workflows follow the ReAct (Reason + Act) framework: observe, plan, act, and loop. A single iteration involves three stages:
1. **Tool prediction** -- selecting which tool to call and with what arguments.
2. **Tool execution** -- running the selected tool and obtaining its output.
3. **Response synthesis** -- interpreting the tool output and producing a response (or continuing the loop).

Evaluating agents is harder than evaluating simple text generation because errors can occur at any of these stages, and the causes and remedies differ for each.

### 9.2 Tool Prediction Errors

Four distinct failure modes can occur when the model decides which tool to call and with what arguments.

**Failure mode 1 -- Not using a tool when one is needed.** The user asks "Find a bear near me," the appropriate tool exists, but the model skips it and responds directly -- often with a punt ("Sorry, I cannot do that"). Two potential causes:

- *Tool router error (recall failure):* If a tool router or selector is used to filter down the available tools before injecting them into the prompt, the relevant tool may have been filtered out. The fix is to adjust the tool router to improve recall for this query type.
- *Model not recognizing the tool use pattern:* The tool was included in the prompt, but the model chose to respond directly instead of calling it. The fix depends on how the model was taught to use tools -- either revisit the SFT training data to include more examples of this pattern, or refine the prompt engineering to make tool use more obvious.

**Failure mode 2 -- Hallucinating a tool that does not exist.** The model calls `find_bear()` when the actual API is `find_teddy_bear()`. The model invents a plausible-sounding function name that was never defined. Potential causes and remedies:

- *Model too weak:* A model with insufficient capacity may fail to ground on the provided instructions and instead generate plausible-looking but incorrect function names. Upgrading to a more capable model may resolve this.
- *Poor API naming:* If the function names, argument names, and docstrings are unclear or inconsistent, the model has less to anchor on. Rewriting the API with clear, descriptive names and thorough documentation is often the most effective fix.
- *Unclear horizontal instructions:* The system-level instructions may not clearly state that the model must use only the provided functions. Making this constraint explicit and prominent in the preamble reduces hallucinated tool calls.

**Failure mode 3 -- Calling the wrong tool.** The right category of action is taken, but the wrong specific tool is selected. For example, sending a message asking about bears instead of calling the bear-finder tool. This confusion arises when tool scopes overlap or are ambiguously defined. The fix: ensure each tool's API description clearly delineates its scope, and verify that the tool router surfaces the correct tools for the query type.

**Failure mode 4 -- Right tool, wrong arguments.** The model calls the correct function but with incorrect parameters -- for example, passing coordinates (0, 0) because the user's location was never provided. This may indicate missing context (the location was not in the prompt or settings) or poor argument documentation. Remedies include adding prerequisite tools (a location-finder that runs first), surfacing actionable errors when required information is unavailable, and improving argument descriptions in the API definition.

### 9.3 Tool Execution Errors

Even when the correct tool is called with the correct arguments, the tool itself may fail.

**Failure mode 5 -- Tool returns incorrect or error output.** The tool has a bug and raises an exception or returns wrong data. When the model receives an error traceback, it often produces an unhelpful response like "Sorry, I encountered an error" without actionable information for the user. The fix is twofold: fix the bug in the tool implementation, and design tool outputs to return structured, meaningful error information rather than raw exceptions.

**Failure mode 6 -- Tool returns no output.** This is particularly problematic for action-performing tools. If a tool increases a thermostat but returns nothing, the model has no confirmation signal and may falsely tell the user the action succeeded (or failed). The guidance is: **always return meaningful output from tool calls**, even when the result is empty. An empty JSON object `{}` is semantically meaningful ("no results found"), while `None` is ambiguous. For action tools, return an explicit success/failure status with relevant details.

### 9.4 Response Synthesis Errors

**Failure mode 7 -- Model fails to incorporate tool output.** The tool returns valid results (e.g., a bear named "Teddy" located one mile away), but the model responds "I didn't find any bear." Possible causes:

- *Model grounding weakness:* The model fails to attend to and incorporate the tool output in its response. This was more common in earlier LLM iterations and is less frequent with modern models.
- *Tool output too verbose:* The relevant information is buried in a large output payload. The model cannot distinguish the signal from the noise. The fix: trim tool outputs to include only the information the model needs for the next step.
- *Poor output formatting:* Raw, unstructured data is harder for the model to parse than well-structured objects with named attributes. Returning typed objects (e.g., `TeddyBear(name="Teddy", distance_miles=1.0)`) rather than raw dictionaries or text blobs improves grounding.

### 9.5 Common Remediation Themes

Across all seven failure modes, the fixes cluster into two categories:

**Modeling improvements:** Upgrade the model's reasoning and grounding capabilities, improve the tool router's recall, refine the SFT training data for tool use, or adjust prompt instructions.

**Tool engineering improvements:** Fix bugs in tool implementations, improve API naming and documentation, ensure meaningful outputs, and trim output payloads to only relevant information.

The overarching guidance for agent debugging is to be **methodical** -- categorize errors into these buckets, address them in groups, and avoid treating each failure as an isolated incident.

---

## 10. Benchmarks

Benchmarks provide standardized tasks and metrics for comparing LLMs against each other and tracking progress over time. They fall into several categories based on what capability they test.

### 10.1 Knowledge Benchmarks -- MMLU

**MMLU** (Massive Multitask Language Understanding) tests whether a model can recall and apply factual knowledge across a broad range of domains. It contains roughly 60 different tasks spanning subjects like law, medicine, history, physics, and everyday knowledge.

The format is multiple-choice with four options per question, making evaluation deterministic -- the model outputs a letter, and we check if it matches the answer key. This avoids the noise of free-form evaluation.

MMLU primarily measures the quality of **pretraining** -- how effectively the model retained information from its training corpus. Example: a medical question might present patient vitals and ask where the damage is located, requiring domain-specific knowledge that pure reasoning cannot derive.

The multiple-choice format is a deliberate design choice shared by many benchmarks. Free-form answers would require an LLM judge for evaluation, introducing another layer of potential error. Constrained formats allow hard-coded answer extraction and comparison.

### 10.2 Reasoning Benchmarks -- AIME and PIQA

**AIME** (American Invitational Mathematics Examination) is a real high school math competition used as an Olympiad qualifier. Problems require multi-step mathematical reasoning and produce a three-digit integer answer, making evaluation straightforward (the answer is either correct or not).

AIME problems are deceptively compact -- a single-sentence problem statement may require extended reasoning to solve. This makes it a strong test of a model's chain-of-thought capabilities. New AIME exams are administered each year, providing fresh problems that the model provably has not seen during training.

**PIQA** (Physical Interaction Question Answering) tests common-sense reasoning grounded in the physical world. It presents everyday scenarios with two possible solutions, and the model must choose the correct one. With 20,000 examples and a binary-choice format, it provides broad coverage.

A representative example: "How do I find something I lost on the carpet?" Option 1: "Vacuum with a solid seal." Option 2: "Vacuum with a hairnet." The correct answer (hairnet) requires understanding that a solid seal would suck the item into the vacuum while a hairnet would trap it at the nozzle. This is physical intuition, not formal knowledge.

### 10.3 Coding Benchmarks -- SWE-bench

**SWE-bench** (likely "Software Engineering Benchmark") tests a model's ability to solve real software engineering problems. The benchmark was constructed by mining popular Python repositories for pull requests that (a) fixed a reported issue and (b) introduced tests. The key insight: if a PR introduces tests alongside a fix, those tests presumably fail without the fix and pass with it.

The evaluation pipeline:
1. Present the model with a GitHub issue and the associated codebase.
2. The model produces a **patch** (a code diff).
3. Apply the patch to the codebase.
4. Run the tests introduced in the original PR.
5. If the tests pass, the model's fix is correct.

This is a form of **test-driven evaluation** -- the tests serve as an automated, objective oracle. No LLM judge or human rater is needed.

SWE-bench is relevant beyond coding use cases. Agentic workflows rely on the model's ability to read and write code (for tool calls, API interactions, etc.), so coding proficiency is a prerequisite for strong agent performance.

### 10.4 Safety Benchmarks -- HarmBench

**HarmBench** (Harmful Behavior Benchmark) evaluates whether a model can be induced to produce harmful content. It has four categories:
- **Standard:** Direct attempts to elicit harmful behavior.
- **Copyright:** Testing whether the model generates copyrighted content.
- **Contextual:** Harmful behavior triggered by specific textual context.
- **Multimodal:** Harmful behavior involving non-text modalities.

A critical design choice: HarmBench distinguishes model quality from safety. If the model attempts to comply with a harmful request -- even if the output is low quality or incomplete -- the attack is counted as successful. The bar is intent, not competence.

Because harmful outputs are open-ended and cannot be evaluated by simple regex matching, HarmBench uses a **trained classifier** to determine whether an attack succeeded. This is the only benchmark discussed in the lecture that relies on a learned evaluator rather than hard-coded answer matching, making it more susceptible to evaluation error.

Safety benchmarks carry a caveat: each LLM provider has its own safety policies, and these policies are not universal. A benchmark score only has meaning relative to a specific policy framework. Model cards typically include a safety section describing the provider's approach, but direct cross-model comparisons on safety benchmarks can be misleading.

### 10.5 Agent Benchmarks -- tau-bench and pass-hat-k

**tau-bench** (Tool Agent User benchmark) evaluates end-to-end agent performance across two domains: airline customer service and retail. It provides:
- A set of **tools** the agent can call.
- A set of **policies** constraining what the agent can and cannot do.
- A set of **tasks** representing user goals (e.g., changing a flight).

A distinguishing feature is that the user side of the conversation is **simulated by a separate LLM**. This is necessary because the user's responses depend on the agent's actions -- the conversation cannot be hard-coded. A large model plays the role of the user, responding dynamically to the agent's actions.

Evaluation checks two things: (1) whether the database state matches the expected outcome (e.g., the flight was actually changed in the system), and (2) whether the correct actions were performed.

tau-bench introduces the metric **pass-hat-k** ($\hat{\text{pass}}@k$), which is the probability that **all** $k$ attempts at a task succeed:

$$\hat{\text{pass}}@k = \left(\frac{c}{n}\right)^k$$

where $c$ is the number of successful attempts out of $n$ total attempts. This contrasts with the more common **pass@k** from Lecture 7, which measures the probability that **at least one** of $k$ attempts succeeds. The distinction matters because agent tasks in domains like airline booking and retail require **reliability** -- a tool that succeeds 70% of the time is not acceptable for production deployment. pass-hat-k captures this reliability requirement by penalizing inconsistency.

---

## 11. Benchmarks in Practice

### 11.1 Real-World Model Comparisons

When frontier model providers (Google, OpenAI, Anthropic) release new models, they publish benchmark results using the categories discussed above -- often with domain-specific variants. For instance, Google's Gemini launch report included:
- **Global PIQA** (a multilingual extension of PIQA) for reasoning.
- A flavor of **SWE-bench** for coding.
- **tau-squared-bench** (an extension of tau-bench) for tool use.

These benchmark suites characterize a model's **profile** -- its strengths and weaknesses across different capabilities. No model dominates every benchmark. Practical model selection depends on your specific use case.

### 11.2 The Pareto Frontier

Model comparison becomes more nuanced when you factor in non-quality dimensions like cost, latency, and context length. For a given quality metric, you can plot models on a price-vs-performance graph. The models that are not dominated (no other model is both cheaper and better) form the **Pareto frontier**. Different applications will choose different points on this frontier depending on their budget and quality requirements.

### 11.3 Data Contamination

A benchmark is only valid under the assumption that the model has not seen the test data during training. If benchmark questions or answers leak into the training corpus, performance numbers become meaningless -- the model is recalling, not reasoning.

Mitigation strategies include:
- **Hash-based detection:** Compute hashes of benchmark items and check for matches in training data.
- **Blocklists:** For tool-use benchmarks, block access to websites that might contain answers.
- **Fresh test sets:** For math benchmarks like AIME, new exams are administered annually, guaranteeing the model could not have trained on them.

### 11.4 Goodhart's Law

"When a measure becomes a target, it ceases to be a good measure."

This adage is directly applicable to LLM benchmarks. If model development optimizes specifically for benchmark performance, the benchmark scores may improve without corresponding improvements in real-world utility. Benchmark scores should inform model selection, not define it.

Counterbalances include broad benchmark suites (harder to game many benchmarks simultaneously), human preference evaluations like **ChatBot Arena** (where real users compare model outputs in blind head-to-head tests), and ultimately, direct testing on your own use case. No benchmark substitutes for trying the model on your actual workload.

---

## Key Takeaways

1. **Human evaluation is the gold standard but does not scale.** It is expensive, slow, and subject to inter-rater disagreement. Metrics like Cohen's kappa, Fleiss's kappa, and Krippendorff's alpha quantify agreement relative to chance, providing a health check on rating consistency.

2. **Rule-based metrics (METEOR, BLEU, ROUGE) compare outputs to references via n-gram overlap.** They are fast and cheap but fundamentally limited: they cannot handle stylistic variation, correlate weakly with human judgment, and still require human-written references.

3. **LLM-as-a-Judge is the current best practice for scalable evaluation.** It requires no reference text, produces interpretable rationales alongside scores, and supports both pointwise and pairwise evaluation modes. Structured output ensures reliable parsing of judge responses.

4. **LLM judges have systematic biases** -- position bias (favoring the first response), verbosity bias (favoring longer responses), and self-enhancement bias (favoring outputs from the same or similar models). Each has specific mitigation strategies.

5. **Factuality evaluation requires decomposition.** Break text into atomic facts, verify each independently (using RAG or web search), then aggregate with optional importance weighting. This captures the nuance that a response can be partially correct.

6. **Agent evaluation spans seven failure modes** across three stages (tool prediction, tool execution, response synthesis). Systematic categorization of errors into modeling vs. tool engineering issues is essential for effective debugging.

7. **Benchmarks characterize model profiles across knowledge (MMLU), reasoning (AIME, PIQA), coding (SWE-bench), safety (HarmBench), and agent tasks (tau-bench).** Most use constrained answer formats to enable hard-coded evaluation, avoiding the need for an LLM judge.

8. **Data contamination and Goodhart's law are persistent threats to benchmark validity.** Fresh test sets, hash-based detection, and real-world user evaluations (like ChatBot Arena) serve as counterweights. Ultimately, the best evaluation is testing the model on your own workload.
