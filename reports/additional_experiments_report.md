# Additional Experiments — WI-IAT 2026 Revision

Four supplementary analyses specified in `specs/additional_experiments_spec.md`,
run to fill the remaining ~0.5 page. All numbers below are reproducible via:

```
uv run python -m dtom.additional_experiments --option all
```

Outputs: `output/additional_option{1,2,3,4}_*.json`, `output/additional_option2_lifts.csv`.
Source code: `src/dtom/additional_experiments.py`.

| Option | Question | Headline result | Feasible? |
|---|---|---|---|
| 1 | *Where* do rule-based & consensus disagree? | Rule-based **A** splits ~evenly: 17.8% → B, **21.3% → C** | ✅ existing files |
| 2 | *Why* does GPT-4o lift A→C? | Corroborated lifts **81% genuine depth**; reverted lifts **100% over-reads** | ✅ existing files |
| 3 | Is within-category depth specific to *Press for Accuracy*? | **Yes** — Revoicing/Restating are ~99% Surface | ✅ corpus re-run |
| 4 | Does depth scale with grade? | **No — it *declines*** (ρ = −0.30, p < .001) | ✅ Subset-1 metadata |

---

## Option 1 — Confusion matrix (rule-based × consensus)

**Table O1. Three-level confusion matrix, rule-based (rows) × two-of-three LLM consensus (cols), n = 200.**

| Rule \ Consensus | A | B | C | Total |
|---|---|---|---|---|
| **A** | 106 | 31 | 37 | 174 |
| **B** | 3 | 16 | 1 | 20 |
| **C** | 0 | 2 | 4 | 6 |
| **Total** | 109 | 49 | 42 | 200 |

Exact three-level agreement = **126/200 (63.0%)** — the diagonal, and identical to the rule-based↔consensus agreement reported in Study 3 (§3.2).

**Finding.** Of the 174 utterances the rule-based classifier called Surface (A), the consensus keeps **106 (60.9%)** at Surface, lifts **31 (17.8%)** to Intermediate (B), and lifts **37 (21.3%)** to Deep (C). Contrary to the spec's prior hypothesis that disagreement would concentrate in the A→B (procedural) transition, the larger off-diagonal mass is **A→C (Deep)**. This is fully consistent with the Study-3 finding that the LLMs *over-extend the deepest level*: even a conservative majority-vote consensus relocates more rule-based-Surface utterances to Deep than to Intermediate. The rule-based B and C rows are too small (20, 6) to disagree much, so the κ structure is driven almost entirely by the A row.

> ⚠️ Note vs. spec draft: the spec's illustrative sentence ("disagreement concentrated in the A→B transition") is **not** supported. Report the A→C dominance instead — it strengthens, rather than weakens, the over-extension narrative.

---

## Option 2 — Why GPT-4o lifts A→C (qualitative reason analysis)

GPT-4o produced **67 A→C lifts** (rule-based = Surface, GPT-4o = Deep). Claude independently adjudicated each: **47 corroborated** (Claude also non-Surface) and **20 reverted** (Claude back to Surface). We hand-coded each GPT-4o reason into one of four types (rubric below); the per-utterance coding is in `output/additional_option2_coding.json`, the evidence in `output/additional_option2_lifts.csv`.

**Table O2. Reason type for GPT-4o's A→C lifts, corroborated vs. reverted.**

| Reason type | Corroborated (n=47) | Reverted (n=20) |
|---|---|---|
| Implicit reasoning demand | **31 (66%)** | 0 (0%) |
| Contextual probing | 7 (15%) | 0 (0%) |
| Pragmatic reinterpretation | 8 (17%) | **13 (65%)** |
| Keyword over-extension | 1 (2%) | **7 (35%)** |

**Rubric.**
- **Implicit reasoning demand** — the utterance text itself carries a conceptual *why/how/meaning* demand the rule-based keywords miss (*"How do we measure area?"*, *"What is the need to simplify?"*, *"What is that comma saying?"*).
- **Contextual probing** — an elliptical utterance (*"What do you mean?"*, *"Wyatt, what are you thinking?"*) that is deep **only** because the preceding turns supply a specific student idea to probe.
- **Pragmatic reinterpretation** — a directive/procedural/factual utterance re-read as a reasoning invitation (*"What is your estimate?"*, *"We need to look at this left-over piece…"*).
- **Keyword over-extension** — a bare solicitation triggered by a mental-state word with no genuine depth (*"What do you think?"*, *"What do we know about volume?"*).

**Finding.** The two piles separate almost perfectly. **Corroborated lifts are 81% genuine depth** (66% implicit reasoning demand + 15% contextual probing): the rule-based classifier misses these mainly because the conceptual demand is phrased *without* its trigger keywords, not because depth always requires conversational context. **Reverted lifts are 100% over-reads** (65% pragmatic reinterpretation of procedural/factual prompts + 35% keyword over-extension on "think"/"know"), with **zero** genuine-depth cases. This is the mechanism behind Study 3's headline: context-aware reading recovers real missed depth (the conservative-lower-bound story) *and* over-extends — and the second LLM (Claude) reliably separates the two, reverting exactly the keyword/pragmatic over-reads.

---

## Option 3 — Within-category depth across all Level-1 categories

The same 35-pattern classifier was applied to all three Level-1 talk-move categories (utterances > 3 words).

**Table O3. Within-category mentalizing depth by Level-1 category (full TalkMoves corpus).**

| Category | N | Surface | Interm. | Deep | Evidence gradient (A→B→C) | χ² | p |
|---|---|---|---|---|---|---|---|
| Press for Accuracy | 20,770 | 87.8% | 9.6% | 2.6% | 7.4% → 26.0% → 29.8% | 714.2 | <.001 |
| Revoicing | 3,093 | 99.0% | 0.5% | 0.5% | 9.2% → 33.3% → 0.0% | 6.8 | .033 |
| Restating | 390 | 97.7% | 1.5% | 0.8% | 10.1% → 0.0% → 0.0% | 0.6 | .757 |

**Finding.** Within-category depth variation is **specific to Press for Accuracy**. Revoicing (99.0% Surface) and Restating (97.7% Surface) are near-degenerate: they contain almost no non-surface variation to detect, the evidence gradient is flat/non-monotonic on tiny cells (Revoicing B n=9, C n=6; Restating B n=6, C n=3), and only Press for Accuracy shows a strong, monotonic, well-powered gradient (χ²=714, V=0.215). This is the expected and theoretically coherent result: Revoicing and Restating are *by definition* acknowledgment moves (the teacher echoes/repeats a student contribution), so they have little room for depth variation, whereas Press for Accuracy is a *questioning* category whose surface form ("checking an answer") can conceal anything from a number-check to a conceptual probe. The phenomenon the paper documents is therefore not an artifact of one classifier on one category — it is concentrated exactly where the construct predicts it should be.

---

## Option 4 — Depth distribution by grade level

Grade metadata is available for the **184 Subset-1 transcripts** in the TalkMoves *Datasheet for Public Release* (the 376 Subset-2 "talkback" transcripts carry no grade label). L3 depth uses the Study-1 talk-move→depth map (Deep L3 = PressReasoning + GettingStudentsRelate).

**Table O4. L3 depth by grade band (Subset-1, 184 transcripts).**

| Grade | Transcripts | N utt. | % Deep L3 | Mean depth | Evidence rate (Deep) |
|---|---|---|---|---|---|
| 3–4 | 84 | 8,700 | **9.0%** | 0.367 | 30.0% |
| 5–6 | 37 | 1,991 | 3.7% | 0.272 | 18.8% |
| 7–8 | 51 | 19,549 | 1.3% | 0.295 | 30.5% |
| 9–12 | 12 | 1,255 | 2.7% | 0.208 | 18.8% |

Transcript-level trend: Spearman **ρ = −0.303, p < .001** (n = 184 transcripts) between ordinal grade band and mean L3 depth.

**Finding.** Deep mentalising does **not** increase with grade — it **declines**. Deep L3 moves are far more frequent in grades 3–4 (9.0%) than in any higher band (≤3.7%), and transcript mean depth falls significantly across grade bands (ρ = −0.30). This contradicts the "higher grades → more abstraction → more depth" hypothesis and supports the alternative: within this corpus, mentalising depth tracks **teacher/pedagogical style in the early grades** rather than content complexity. The implication for AI classifiers is the reverse of the intuitive prior — calibration cannot assume deep mentalising is a high-grade phenomenon.

> ⚠️ Caveats to state in-text: (i) grade coverage is Subset-1 only (184/566 transcripts); (ii) the 9–12 band is thin (12 transcripts); (iii) the per-band "evidence rate (Deep)" is non-monotonic across grades, so this is a **distributional** finding about depth frequency, not a claim about deep-move *effectiveness* by grade.

---

## Recommended inclusion

For ~0.5 page, **Options 1 + 2** remain the strongest pairing (no new data runs, directly extends Study 3, answers "where" and "why" the classifiers diverge). **Option 3** is the best single add if a generalisability paragraph for RQ1 is wanted (it cleanly bounds the phenomenon to Press for Accuracy). **Option 4** is feasible and produces a counter-intuitive, citable result, but needs the three caveats above and ~a third of a page once stated honestly.

---

## Ready-to-paste LaTeX

```latex
% ---------- Option 1 ----------
\begin{table}[t]
\centering
\caption{Three-level confusion matrix: rule-based (rows) vs.\ two-of-three
LLM consensus (columns), $n=200$. Exact agreement $=63.0\%$.}
\label{tab:confusion}
\begin{tabular}{lcccc}
\toprule
Rule $\backslash$ Consensus & A & B & C & Total \\
\midrule
A & 106 & 31 & 37 & 174 \\
B & 3 & 16 & 1 & 20 \\
C & 0 & 2 & 4 & 6 \\
\midrule
Total & 109 & 49 & 42 & 200 \\
\bottomrule
\end{tabular}
\end{table}

% ---------- Option 2 ----------
\begin{table}[t]
\centering
\caption{Reason type for GPT-4o's 67 surface$\rightarrow$deep (A$\rightarrow$C)
lifts, split by Claude's adjudication.}
\label{tab:reasons}
\begin{tabular}{lcc}
\toprule
Reason type & Corroborated (47) & Reverted (20) \\
\midrule
Implicit reasoning demand & 31 (66\%) & 0 (0\%) \\
Contextual probing & 7 (15\%) & 0 (0\%) \\
Pragmatic reinterpretation & 8 (17\%) & 13 (65\%) \\
Keyword over-extension & 1 (2\%) & 7 (35\%) \\
\bottomrule
\end{tabular}
\end{table}

% ---------- Option 3 ----------
\begin{table*}[t]
\centering
\caption{Within-category mentalizing depth across all three Level-1 talk-move
categories (full TalkMoves corpus, utterances $>3$ words).}
\label{tab:withincat}
\begin{tabular}{lrcccccr}
\toprule
Category & $N$ & Surface & Interm. & Deep & Evidence gradient (A$\to$B$\to$C) & $\chi^2$ & $p$ \\
\midrule
Press for Accuracy & 20{,}770 & 87.8\% & 9.6\% & 2.6\% & 7.4\%$\to$26.0\%$\to$29.8\% & 714.2 & $<.001$ \\
Revoicing & 3{,}093 & 99.0\% & 0.5\% & 0.5\% & 9.2\%$\to$33.3\%$\to$0.0\% & 6.8 & .033 \\
Restating & 390 & 97.7\% & 1.5\% & 0.8\% & 10.1\%$\to$0.0\%$\to$0.0\% & 0.6 & .757 \\
\bottomrule
\end{tabular}
\end{table*}

% ---------- Option 4 ----------
\begin{table}[t]
\centering
\caption{L3 mentalizing depth by grade band (TalkMoves Subset-1, 184
transcripts with grade metadata). Spearman $\rho=-0.30$, $p<.001$.}
\label{tab:grade}
\begin{tabular}{lrrcc}
\toprule
Grade & Transcripts & $N$ utt. & \% Deep L3 & Mean depth \\
\midrule
3--4 & 84 & 8{,}700 & 9.0\% & 0.367 \\
5--6 & 37 & 1{,}991 & 3.7\% & 0.272 \\
7--8 & 51 & 19{,}549 & 1.3\% & 0.295 \\
9--12 & 12 & 1{,}255 & 2.7\% & 0.208 \\
\bottomrule
\end{tabular}
\end{table}
```

---

*Provenance: Options 1–2 from `output/llm_consensus.csv`, `llm_coding_results.json`,
`llm_coding_results_claude.json`. Options 3–4 from a fresh rule-based pass over the
TalkMoves corpus (567 transcripts, 237,537 utterances) and
`data/TalkMoves/data/Datasheet for Public Release.xlsx`. Prepared 2026-06-14.*
