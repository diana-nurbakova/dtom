# Data Licensing

This repository mixes two licences. The source code is released under the MIT License (see [LICENSE](LICENSE)). Files derived from third-party datasets inherit the terms of those datasets, as set out below.

## TalkMoves-derived files: CC BY-NC-SA 4.0

The following files contain transcripts, utterances, annotations, or results derived from the **TalkMoves dataset** (Suresh et al., 2022) and are therefore licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0), the licence of the original dataset:

| Path | Content |
|------|---------|
| `dtom-lens/data/sample_transcripts/` | Verbatim TalkMoves transcripts |
| `dtom-lens/data/precomputed/talkmoves_results.json` | Results computed on TalkMoves, including example utterances |
| `dtom-lens/data/precomputed/transcript_l3_analysis.csv` | Transcript-level results on TalkMoves |
| `dtom-lens/data/precomputed/llm_agreement.json` | LLM-classifier results on TalkMoves utterances |
| `dtom-lens/data/precomputed/llm_convergence.json` | LLM-classifier results on TalkMoves utterances |
| `output/` | All TalkMoves results, sampled utterances, LLM codings, and figures |
| `reports/` | Analysis reports, which quote TalkMoves utterances |

Under this licence you may share and adapt these files for **non-commercial** purposes, provided you give appropriate credit and distribute any derivative work under the same licence. Please credit the original dataset:

> Suresh, A., Jacobs, J., Clevenger, C., Lai, V., Tan, C., Ward, W., Martin, J. H., & Sumner, T. (2022). The TalkMoves Dataset: K-12 Mathematics Lesson Transcripts Annotated for Teacher and Student Discursive Moves. *LREC 2022*.
>
> TalkMoves dataset: https://github.com/SumnerLab/TalkMoves

and this repository:

> Nurbakova, D., & Baig, D. (2026). Mentalising Depth in Teacher Discourse: What AI Classifiers Collapse and What It Predicts. WI-IAT 2026.

## NCTE: not redistributed

No data from the **NCTE Classroom Transcript Dataset** (Demszky & Hill, 2022) is redistributed in this repository. The NCTE dataset is available only under a data use agreement, via individual application (see the [NCTE repository](https://github.com/ddemszky/classroom-transcript-analysis)).

The NCTE-related files in this repository (`ncte_output/ncte_replication_results.json` and `dtom-lens/data/precomputed/ncte_results.json`) contain only **aggregate statistics** computed by the authors (counts, percentages, test statistics). They contain no transcripts, utterances, or annotations. To reproduce them, obtain NCTE access yourself and place the files in `data/NCTE/` (git-ignored).

## Everything else: MIT

All other files, including the source code in `src/`, `main.py`, and the `dtom-lens/` application code, are released under the MIT License.
