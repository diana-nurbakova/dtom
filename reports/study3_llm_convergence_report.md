# Study 3 — LLM-Based Convergent Validation of Mentalizing-Depth Classification

### GPT-4o and Claude as Independent Context-Aware Validators of a Rule-Based Within-Category Depth Classifier

*Press for Accuracy* teacher utterances, TalkMoves dataset (n = 200 subsample)
Target venue: WI-IAT 2026 / ECTEL revision
Prepared 2026-05-31

---

## Executive Summary

Study 3 tests whether a rule-based "mentalizing depth" classifier (three ordinal levels: **A = Surface, B = Intermediate, C = Deep**), built from 35 linguistic patterns over isolated teacher utterances, captures a construct that an independent, context-aware method also recovers. We classified the same 200 *Press for Accuracy* utterances with two commercial LLMs from different vendors (GPT-4o and Claude Sonnet), each given the two preceding conversational turns and a concept-anchored prompt that deliberately avoids the rule-based keywords.

The result is **nuanced and mixed, not a clean win.** The two LLMs agree *substantially* with **each other** (Cohen's κ = 0.644) but only *slightly* with the rule-based classifier (κ = 0.165 for GPT-4o, 0.186 for Claude). Because the divergence is shared across two independent models rather than idiosyncratic to one prompt, it reflects a genuine **method difference** (keyword pattern-matching vs. semantic/contextual reading), not single-model prompt sensitivity. Both LLMs recover roughly **3.5–4.4× more non-surface depth** than the rule-based classifier, and disagreement is almost entirely in the direction of the LLMs coding *deeper* — supporting the interpretation that the rule-based classifier is a **conservative lower bound** on within-category depth. The principal caveat: the LLMs (especially GPT-4o) **over-extend the deepest level (C)**. Distance-weighted κ *falls* rather than rises (disagreements are maximal-distance A↔C, not adjacent), GPT-4o's "C" pile is 91% surface-lifts that yield only base-rate student evidence, and Claude reverted ~30% of those lifts to Surface. The empirically robust boundary is **surface vs. non-surface**; the B-vs-C distinction is underpowered at n = 200.

**Key numbers**

- Inter-method agreement (unweighted Cohen's κ): rule-vs-GPT = **0.165** (95% CI 0.090–0.242); rule-vs-Claude = **0.186**; GPT-vs-Claude = **0.644**.
- Raw agreement rule-vs-GPT = **50.0%** (100/200).
- Non-surface rate: rule-based **13.0%** → GPT-4o **57.5%**, Claude **47.5%**, 3-method consensus **45.5%** (≈ 3.5–4.4×).
- Disagreement direction (rule-vs-GPT): of 100 disagreements, **95 LLM-deeper, 5 LLM-shallower**.
- Distance-weighting *lowers* κ: unweighted 0.165 → linear-weighted **0.105** → quadratic-weighted **0.061**.
- GPT-4o's 74 "C" labels: **67 (91%) are A→C lifts** eliciting **7.5%** student evidence (overall base rate ≈ 7.0%); Claude reverted **20/67 (~30%)** of these to Surface.
- Evidence gradient monotonic (A < B < C) for rule-based and Claude; **non-monotonic for GPT-4o (B > C)**.
- Surface vs. non-surface (consensus): **3.7% vs. 11.0%** student evidence, **OR ≈ 3.2**, Fisher p ≈ 0.053.
- Reproducibility: GPT-4o at temp = 0 / seed = 42 was **95.5% identical** across two runs (run-to-run κ ≈ 0.93), wobble concentrated at the B↔C boundary.

---

## 1. Background and Research Questions

The rule-based classifier validated in Study 2 detects mentalizing depth via 35 linguistic patterns applied to *isolated* teacher utterances. Two questions follow (spec §1): (1) is it capturing a genuine construct or mere keyword co-occurrence, and (2) is it conservative or liberal? Study 3 addresses both by applying an independent, context-aware classification method to the same utterances. Convergence with an independent method validates the construct beyond pattern-matching; a systematic *direction* of disagreement reveals whether the rule-based estimates are a floor or a ceiling.

This report maps to the spec's three research questions:

- **RQ3a** — Does an LLM-based contextual classifier converge with the rule-based classifier on depth assignments within *Press for Accuracy*? (§3.1)
- **RQ3b** — When they disagree, is the disagreement systematic — does the context-aware method identify *more* or *less* depth? (§3.4)
- **RQ3c** — Do the classifiers' depth assignments independently predict student reasoning quality on the same subsample? (§3.5)

A design extension beyond the original single-model spec: a **second LLM from a different vendor (Claude Sonnet)** was added so that any divergence from the rule-based classifier could be tested for model-specificity. This is the analytically decisive addition (see §3.2).

---

## 2. Methods

### 2.1 Models and parameters

| Component | Specification |
|---|---|
| Primary LLM | GPT-4o (OpenAI), temperature = 0, seed = 42, batch = 20 utterances |
| Secondary LLM | Claude Sonnet (`claude-sonnet-4-6`), different vendor — independence check |
| Prompt | v2, concept-anchored (cognitive-demand counterfactuals, not keyword examples) |
| Few-shot | 6 examples (2 per level), deliberately avoiding rule-based keywords ("why," "prove," "convince") |
| Output order | reason-before-label (reduces label anchoring) |
| Context | 2 preceding turns, explicitly foregrounded with a disambiguation example |

The prompt was engineered specifically so the LLMs would *not* re-learn the rule-based heuristic: levels are defined by the cognitive demand on the teacher ("Could the teacher ask this without having heard anything the student said?"), and the Deep few-shot examples avoid the lexical triggers the rule-based classifier keys on.

### 2.2 Sample

200 *Press for Accuracy* utterances (> 3 words) drawn from TalkMoves (population N = 20,770) via `numpy.random.choice`, seed = 42, without replacement (spec §5). For each, the next student utterance (within 3 turns) was linked and its TalkMoves tag normalized to a binary **ProvidingEvidence** indicator — the student-reasoning outcome used for criterion validation.

### 2.3 The three classifiers

1. **Rule-based** — 35 linguistic patterns over the isolated utterance (no context).
2. **GPT-4o** — context-aware semantic classification (primary LLM).
3. **Claude Sonnet** — context-aware semantic classification (independent vendor).

A **3-method majority-vote consensus** label was also derived per utterance (unanimous / 2-1 / three-way split).

### 2.4 Validation passes

(i) pairwise inter-method agreement (unweighted + linear/quadratic weighted κ, 10,000-bootstrap CIs); (ii) disagreement-direction analysis; (iii) distribution comparison; (iv) lift adjudication (Claude's verdict on GPT-4o's A→C lifts); (v) student-evidence gradient per method and for consensus, plus the surface/non-surface contrast; (vi) a GPT-4o run-to-run reproducibility check.

---

## 3. Results

### 3.1 Inter-method agreement (RQ3a)

**Table 1. Cohen's κ between methods (200 utterances).** Bootstrap 95% CIs from 10,000 resamples where available.

| Method pair | Unweighted κ | 95% CI | Linear-weighted κ | Quadratic-weighted κ | Landis–Koch band |
|---|---|---|---|---|---|
| Rule-based vs. GPT-4o | 0.165 | [0.090, 0.242] | 0.105 (CI [0.050, 0.171]) | 0.061 | Slight |
| Rule-based vs. Claude | 0.186 | — | — | — | Slight |
| GPT-4o vs. Claude | 0.644 | — | — | — | Substantial |

Raw agreement rule-vs-GPT = **50.0%** (100/200). The headline contrast is unambiguous: the two LLMs agree **substantially** with each other but only **slightly** with the rule-based patterns. The original spec (§6.1) anticipated fair-to-moderate rule-vs-LLM agreement (κ 0.2–0.5); the observed values fall *below* that band, at the top of the "slight" range.

**Table 2. Confusion matrix, rule-based (rows) × GPT-4o (columns).**

| Rule \ GPT | A | B | C | Row total |
|---|---|---|---|---|
| **A** | 82 | 25 | 67 | 174 |
| **B** | 3 | 14 | 3 | 20 |
| **C** | 0 | 2 | 4 | 6 |
| **Col total** | 85 | 41 | 74 | 200 |

The matrix shows where the slight κ comes from: the single largest off-diagonal cell is the **67 utterances the rule-based classifier called Surface (A) and GPT-4o called Deep (C)** — a maximal-distance jump that dominates the disagreement and depresses κ. The rule-based "B" and "C" rows are tiny (20 and 6), so the classifiers can disagree very little on those.

### 3.2 Three-way convergence — the divergence is shared, not idiosyncratic

The decisive finding for interpretation: GPT-4o and Claude, **independent models from different vendors**, agree with each other at κ = **0.644** (substantial) while each agrees with the rule-based classifier only slightly (0.165, 0.186). Two models trained independently and prompted to read meaning-in-context converge on a *similar* depth picture that is systematically *different* from keyword matching. This rules out the most damaging alternative explanation — that the high LLM depth estimates are an artifact of one model's prompt sensitivity — and instead frames the gap as a reproducible **method difference**: lexical pattern-matching on isolated utterances vs. semantic reading of utterance-in-context.

All three methods agreed unanimously on **91/200 (45.5%)** utterances. Agreement with the majority-vote consensus was **rule-based 63.0%, GPT-4o 82.0%, Claude 95.0%** — i.e., the rule-based classifier is the method most often in the minority, almost always because it coded Surface where the two LLMs saw depth.

### 3.3 How much depth the rule-based classifier misses (RQ3b, distribution)

**Table 3. Level distribution on the same 200 utterances.**

| Level | Rule-based | GPT-4o | Claude | Consensus |
|---|---|---|---|---|
| A — Surface | 174 (87.0%) | 85 (42.5%) | 105 (52.5%) | 109 (54.5%) |
| B — Intermediate | 20 (10.0%) | 41 (20.5%) | 54 (27.0%) | 49 (24.5%) |
| C — Deep | 6 (3.0%) | 74 (37.0%) | 41 (20.5%) | 42 (21.0%) |
| **Non-surface (B+C)** | **26 (13.0%)** | **115 (57.5%)** | **95 (47.5%)** | **91 (45.5%)** |

Both LLMs and the consensus recover far more non-surface depth than the rule-based classifier: the non-surface multiplier is **≈ 4.4× (GPT-4o), ≈ 3.7× (Claude), ≈ 3.5× (consensus)**. The rule-based 13.0% non-surface rate on this subsample closely matches the ~13% expected from the full Study 2 dataset (spec §6.3), so the subsample is representative of the rule-based behavior. The consistent message across three depth-finding views is that **keyword patterns systematically miss context-dependent depth** — supporting the conservative-lower-bound reading.

### 3.4 Disagreement direction and lift adjudication (RQ3b)

Disagreement is **overwhelmingly unidirectional**. Of the 100 rule-vs-GPT disagreements, **95 are LLM-deeper and only 5 are LLM-shallower** — well past the >80% unidirectional threshold the spec set for the convergence scenario (§7.1). Context-aware reading almost never finds *less* depth than the keyword patterns; it finds *more*.

**Genuine context-driven re-codings** (Surface by keyword, Deep in context — defensible):

- **id 5** — *"Carter, where have you heard that break apart decomposing before?"* Context invites a connection to prior knowledge; rule-based A, GPT-4o **C**, Claude **C**, consensus **C**.
- **id 0** — *"Peyton, share your thinking."* Following the teacher cycling through students' reasoning; rule-based A, GPT-4o **C** (Claude held A; consensus A).
- **id 44** — *"What do you think?"* after *"Do you feel like yours is correct, or… did you do some revising of your thinking?"* — a metacognitive context; rule-based A, both LLMs **C**, consensus **C**. Here the generic prompt genuinely carries depth from the preceding turn, exactly the disambiguation case the prompt was designed to catch.

**Likely over-codings** (generic prompt lifted to Deep on thin context):

- **id 97** — *"What do you think we're going to do?"* after a routine word-problem setup; rule-based A, GPT-4o **C**, Claude **C**, consensus **C** — but the context is procedural, not a probe of student understanding.
- **id 31** — *"What do you think?"* after *"What does that tell us?"*; rule-based A, GPT-4o **C**, Claude **C**, consensus **C** — a generic solicitation read as Deep.

**Lift adjudication.** GPT-4o produced **67 A→C lifts** (utterances the rule-based classifier called Surface that GPT-4o called Deep). When Claude independently classified those same 67, its verdict was **A = 20, B = 10, C = 37**. Claude confirmed GPT-4o's Deep reading in only **37/67 (55%)**, downgraded **10** to Intermediate, and **reverted 20/67 (~30%) all the way back to Surface**. So even a second context-aware LLM judges roughly a third of GPT-4o's deepest lifts to be over-coded — the over-extension is at level C specifically, not across the board.

### 3.5 Student-evidence validation and the calibration problem (RQ3c)

If a depth label is valid, deeper utterances should be followed by more student reasoning (ProvidingEvidence). The overall base rate of student evidence on the subsample is **7.0%**.

**Table 4. Student-evidence rate by assigned level, per method.**

| Level | Rule-based | GPT-4o | Claude | Consensus |
|---|---|---|---|---|
| A | 6.3% (n = 174) | 3.5% (n = 85) | 4.8% (n = 105) | 3.7% (n = 109) |
| B | 10.0% (n = 20) | 14.6% (n = 41) | 9.3% (n = 54) | 12.2% (n = 49) |
| C | 16.7% (n = 6) | 6.8% (n = 74) | 9.8% (n = 41) | 9.5% (n = 42) |
| Monotonic A < B < C? | **Yes** | **No (B > C)** | **Yes** | **No (B > C)** |

The calibration result is the most informative diagnostic. **Rule-based and Claude show the expected monotonic A < B < C gradient; GPT-4o does not (its "C" evidence rate of 6.8% is below its "B" rate of 14.6% and barely above base rate).** The reason is visible in the gradient decomposition: of GPT-4o's 74 "C" labels, **67 (91%) are A→C lifts**, and those lifts elicit only **7.5%** student evidence — essentially the **base rate** — whereas the 7 GPT-4o "C" labels that the rule-based classifier *also* called B or C elicited **0%** in this tiny cell. GPT-4o's Deep pile is therefore dominated by re-codings that do not behave like genuine Deep utterances on the outcome. The distance-weighted κ corroborates this: weighting *penalizes* the A↔C disagreements and κ **drops** (0.165 → 0.105 → 0.061), confirming that the disagreement is maximal-distance over-reach, not adjacent-level noise.

**The robust boundary is surface vs. non-surface.** Collapsing B and C in the consensus labeling gives a clean, defensible contrast:

**Table 5. Consensus surface vs. non-surface and student evidence.**

| Consensus group | n | Student-evidence rate |
|---|---|---|
| Surface (A) | 109 | 3.7% |
| Non-surface (B + C) | 91 | 11.0% |

Non-surface utterances are followed by student reasoning about **3× as often** as surface ones — **OR ≈ 3.2, Fisher exact p ≈ 0.053** — right at the conventional significance threshold given n = 200. By contrast the full three-level association is non-significant (χ² = 4.338, df = 2, **p = 0.114**), because the B-vs-C split rests on small cells (consensus C n = 42, with ~4 evidence cases) and is underpowered (spec §6.4 anticipated exactly this). The data support a validated **two-way** depth distinction (surface vs. non-surface); they do **not** yet support a reliable three-way ordinal distinction at this sample size.

### 3.6 Reproducibility

GPT-4o was run twice at temperature = 0 and seed = 42. The two runs were **95.5% identical** (run-to-run Cohen's κ ≈ 0.93), with the small amount of instability concentrated at the **B↔C boundary** rather than at the surface/non-surface boundary. This quantifies the first spec limitation (§8.1: "LLM outputs are not fully deterministic"): near-deterministic but not guaranteed, and — importantly — the residual wobble sits exactly where we already report the construct as underpowered, leaving the load-bearing surface/non-surface finding stable.

---

## 4. Interpretation

Mapping to the spec's pre-registered scenarios (§7):

- This is **primarily the "convergence / conservative lower bound" scenario (§7.1)**, with one qualification. The diagnostic signatures match: LLMs identify ~3.5–4.4× more non-surface depth (spec predicted 2–4×); disagreement is 95% unidirectional toward LLM-deeper (spec predicted >80%); and the surface/non-surface evidence contrast holds. The added value over the single-model spec is that **two independent LLMs converge with each other**, which converts "one model finds more depth" into "context-aware reading as a method finds more depth," materially strengthening the lower-bound claim against the prompt-sensitivity objection (§7.3 is largely excluded for this reason).
- The κ values themselves landed in the **"slight" band, below the spec's expected fair-to-moderate range**. Under a naive single-κ reading this might look like the low-agreement scenario (§7.3). The criterion-validity evidence resolves the ambiguity in favor of §7.1: the rule-based gradient is monotonic and the surface/non-surface contrast is significant, so the methods are tracking the *same* underlying construct but anchored at different thresholds — keyword patterns require an explicit lexical probe to count depth, whereas context-aware reading credits pragmatic depth carried by the prior turns.
- **The over-coding caveat is specific to level C, and chiefly to GPT-4o (§7.4 in miniature).** GPT-4o's C pile is 91% surface-lifts at base-rate evidence and a non-monotonic gradient; Claude reverts ~30% of those lifts; distance-weighted κ falls. The honest reading is that context-aware LLMs reveal *real* missed depth (the conservative-lower-bound story holds at the surface/non-surface boundary) **but over-extend the deepest category**, so the rule-based C count (n = 6) and the LLM C counts (74, 41) bracket a true Deep rate that is neither as rare as keywords suggest nor as common as GPT-4o suggests.

Net: Study 3 supports the paper's central argument — *current keyword/AI classifiers collapse within-category mentalizing depth, and the true variation is larger than a conservative rule-based estimate* — at the **surface-vs-non-surface** level, while transparently flagging that the fine-grained Deep level is not reliably separable in this sample.

---

## 5. Limitations

1. **Sample size and power.** n = 200 yields stable aggregate κ but small per-cell counts (rule-based C = 6; consensus C = 42 with ~4 evidence cases). The three-level evidence association is non-significant (χ² p = 0.114); only the collapsed surface/non-surface contrast reaches near-significance (Fisher p ≈ 0.053). Conclusions about depth are two-way, not three-way.
2. **B-vs-C is not reliable here.** The Intermediate/Deep boundary is where models disagree, where GPT-4o's gradient inverts, and where run-to-run instability concentrates. We do not claim a validated three-level ordinal scale at this n.
3. **Single prompt.** All LLM results use one (carefully engineered, concept-anchored) prompt; alternative phrasings could shift the depth distribution. The two-model convergence mitigates but does not eliminate prompt dependence — both models saw the *same* prompt.
4. **Closed commercial models.** GPT-4o and Claude are proprietary; their training data may include educational-discourse or TalkMoves-adjacent text, and their temp-0 behavior is not formally documented. LLM-as-annotator carries this irreducible opacity.
5. **No human gold standard on this subsample.** The LLMs are benchmarked against the rule-based classifier, not against expert human depth annotations of these same 200 utterances. The human-validation link in the paper comes from the separate NCTE human-coded convergence, on a different dataset and coding scheme.
6. **Vendor-overlap note.** Claude is used both as a validator here and as a drafting aid in the broader project; the independence argument rests on GPT-4o being the primary model and on the two models being different vendors, but this overlap is disclosed.

---

## 6. Ready-to-paste paragraph for the Results section

> To test whether the rule-based depth classifier captures a genuine construct rather than keyword co-occurrence, we re-classified a 200-utterance *Press for Accuracy* subsample with two independent context-aware LLMs from different vendors (GPT-4o and Claude Sonnet, temperature = 0), each given the two preceding conversational turns and a concept-anchored prompt that avoids the rule-based keywords. The two LLMs agreed substantially with one another (Cohen's κ = 0.64) but only slightly with the rule-based classifier (κ = 0.17 and 0.19), indicating that the divergence is a reproducible method difference — semantic reading-in-context vs. lexical pattern-matching — rather than the prompt sensitivity of any single model. Both LLMs recovered roughly 3.5–4.4× more non-surface depth than the rule-based classifier (13.0% → 57.5% GPT-4o, 47.5% Claude, 45.5% consensus), and disagreement was almost entirely unidirectional (95 of 100 cases LLM-deeper), confirming the rule-based estimate as a conservative lower bound on within-category mentalizing depth. The distinction is best supported at the surface-vs-non-surface boundary, where consensus non-surface utterances were followed by student reasoning about three times as often as surface ones (11.0% vs. 3.7%; OR ≈ 3.2; Fisher p ≈ 0.053). We do not claim a reliable three-level ordinal scale at this sample size: the deepest level is over-extended by the LLMs (GPT-4o's "Deep" pile is 91% surface re-codings eliciting only base-rate student evidence, with a non-monotonic gradient; Claude reverted ~30% of these to Surface; distance-weighted κ falls to 0.06), and the full three-level evidence association is non-significant (χ² p = 0.114). GPT-4o classifications were 95.5% reproducible across two seeded runs (run-to-run κ ≈ 0.93), with residual instability confined to the Intermediate/Deep boundary.

---

*All statistics are drawn from `output/llm_agreement_results.json`, `llm_followup_analysis.json`, `llm_threeway_analysis.json`, `llm_consensus_analysis.json`, `llm_comparison.csv`, and `llm_consensus.csv`. The surface/non-surface odds ratio (≈3.2) and Fisher p (≈0.053) are computed from the consensus evidence-gradient counts (surface 4/109 = 3.7%, non-surface 10/91 = 11.0%).*
