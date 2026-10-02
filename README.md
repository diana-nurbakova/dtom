# DToM — Double Theory of Mind Empirical Analysis

This repository supports the paper accepted to **WI-IAT 2026**:

> Diana Nurbakova and Duaa Baig. **Mentalising Depth in Teacher Discourse: What AI Classifiers Collapse and What It Predicts.** *IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT 2026).*

If you use this code or the derived data, please cite the paper (see [Citation](#citation)).

Empirical grounding for the Double Theory of Mind (DToM) framework through secondary analysis of classroom discourse data. The theoretical framework is introduced in a companion paper published at **EC-TEL 2026**:

> Duaa Baig, Diana Nurbakova, Sylvie Calabretto, and Baba Mbaye. **Who Understands Whom? Mutual Theory of Mind as a Unified Framework for Teacher-Facing AI.** *EC-TEL 2026*, Springer ([proceedings volume](https://link.springer.com/book/9783032379818), ISBN 978-3-032-37981-8).

The DToM framework proposes that teacher-facing AI creates a three-layer cognitive structure:

- **L1** — the AI's model of the teacher
- **L2** — the teacher's model of the AI
- **L3** — the teacher's Theory of Mind toward students

This project operationalizes L3 and demonstrates that teacher mentalizing depth is measurable, varies meaningfully, predicts student reasoning quality, and contains within-category variation that standard AI coding schemes miss.

## Studies

| Study | Description | Script |
|-------|-------------|--------|
| **Study 1** | L3 Depth Mapping — maps teacher talk moves to mentalizing depth levels and tests whether depth predicts student reasoning | `src/dtom/analysis_pipeline.py` |
| **Study 2** | Within-Category Analysis — shows that "Press for Accuracy" (a single standard category) contains hidden variation in mentalizing depth | `src/dtom/analysis_pipeline.py` |
| **Study 3** | Convergent Validity — uses an LLM-based classifier (Claude Sonnet 4) to independently validate the within-category depth patterns | `src/dtom/llm_classifier.py` |
| **NCTE R1** | Mentalizing Depth Distribution — replicates Study 2 distribution on NCTE data | `src/dtom/ncte_replication.py` |
| **NCTE R2** | Sequential & Transcript-Level Analysis — replicates Study 1 predictive relationship on NCTE data | `src/dtom/ncte_replication.py` |
| **NCTE R3** | Convergence with Paired Annotations — tests whether rule-based classifier aligns with independent human-coded uptake/focusing annotations | `src/dtom/ncte_replication.py` |

## Datasets

### TalkMoves

**TalkMoves** (Suresh et al., 2022, LREC) — 567 annotated K-12 mathematics classroom transcripts (~237,500 utterances).

- Repository: https://github.com/SumnerLab/TalkMoves
- License: CC BY-NC-SA 4.0

### NCTE

**NCTE Classroom Transcript Dataset** (Demszky & Hill, 2022) — ~1,660 elementary mathematics classroom transcripts (~580,000 utterances).

- Access: Restricted; requires individual application via [Google Form](https://forms.gle/1yWybvsjciqL8Y9p8)
- Paper: arXiv:2211.11772
- The NCTE replication tests generalizability of the mentalizing depth classifier across independently collected datasets, annotation schemes, and research groups

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.

```bash
# Clone this repository
git clone https://github.com/diana-nurbakova/dtom.git
cd dtom

# Install dependencies
uv sync

# Clone the TalkMoves dataset
mkdir -p data
git clone https://github.com/SumnerLab/TalkMoves.git data/TalkMoves
```

For Study 3 (LLM classifier), create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

For the NCTE replication, place the CSV files in `data/NCTE/` after obtaining access (see [Datasets](#ncte) above).

## Usage

```bash
# Run Studies 1 & 2 (no API key required)
uv run python main.py

# Run all three studies
uv run python main.py --with-llm

# Run Study 3 only
uv run python main.py --study3-only

# Run NCTE replication (Studies R1-R3)
uv run python main.py --ncte

# Custom data/output paths
uv run python main.py --data-dir path/to/TalkMoves/data --output-dir results/
uv run python main.py --ncte --ncte-data-dir path/to/NCTE --ncte-output-dir results/ncte/
```

Alternatively, run individual scripts directly:

```bash
uv run dtom-pipeline --data-dir data/TalkMoves/data --output-dir output/
uv run dtom-llm --data-dir data/TalkMoves/data --output-dir output/
uv run dtom-ncte --data-dir data/NCTE --output-dir ncte_output/
```

### Additional experiments (WI-IAT 2026 revision)

```bash
uv run python -m dtom.additional_experiments --option all   # or 1, 2, 3, 4, 5
```

Option 5 is the pattern-ablation sensitivity analysis of Study 2 (`specs/pattern_ablation_spec.md`):

```bash
uv run python -m dtom.additional_experiments --option 5
```

Outputs: `output/additional_option5_ablation.json`, `output/additional_option5_loo.csv`. Seed 42 throughout.

Option 5b extends it to both pattern tiers (`specs/pattern_ablation_spec_v2.md`):

```bash
uv run python -m dtom.additional_experiments --option 5b
```

Outputs: `output/additional_option5b_ablation.json`, `output/additional_option5b_loo.csv`, `output/additional_option5b_deadpatterns.json`. Seed 42.

Option 6 computes the matched transcript-level effect size (`specs/matched_effect_size_spec.md`): a 2 x 2 of IV (category mapping | pattern classifier) x DV (evidence tag | >=10-word response) on TalkMoves, with the NCTE value recomputed by the same function. Cell A must reproduce Study 1 (d = 0.468, N = 536) or the run stops.

```bash
uv run python -m dtom.additional_experiments --option 6
```

Outputs: `output/additional_option6_matched_d.json`, `output/additional_option6_transcripts.csv`. Bootstrap B = 10,000, seed 42.

## Output

### TalkMoves (Studies 1-3)

Results are written to the `output/` directory:

| File | Description |
|------|-------------|
| `dtom_results.json` | All statistical results (Studies 1 & 2) |
| `transcript_l3_analysis.csv` | Transcript-level L3 depth and student reasoning data |
| `figure1_l3_depth_mapping.png` | Study 1 figure (4 panels) |
| `figure2_within_category.png` | Study 2 figure (2 panels) |
| `llm_samples.json` | Sampled utterances sent to the LLM (Study 3) |
| `llm_coding_results.json` | Raw LLM classifications with justifications |
| `llm_agreement_results.json` | Inter-method agreement statistics (Cohen's kappa) |

### NCTE Replication (Studies R1-R3)

Results are written to the `ncte_output/` directory:

| File | Description |
|------|-------------|
| `ncte_replication_results.json` | All statistical results (Studies R1-R3) |

## Project Structure

```
dtom/
├── main.py                    # Entry point for all studies
├── pyproject.toml
├── src/dtom/
│   ├── analysis_pipeline.py   # Studies 1 & 2
│   ├── llm_classifier.py     # Study 3
│   └── ncte_replication.py   # NCTE replication (Studies R1-R3)
├── data/
│   ├── TalkMoves/             # TalkMoves dataset (cloned separately, git-ignored)
│   └── NCTE/                  # NCTE dataset (restricted access, git-ignored)
├── output/                    # TalkMoves results
└── ncte_output/               # NCTE replication results
```

## Citation

```bibtex
@inproceedings{nurbakova2026mentalising,
  title     = {Mentalising Depth in Teacher Discourse: What {AI} Classifiers Collapse and What It Predicts},
  author    = {Nurbakova, Diana and Baig, Duaa},
  booktitle = {Proceedings of the IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT 2026)},
  year      = {2026}
}
```

For the underlying theoretical framework, please also cite:

```bibtex
@inproceedings{baig2026who,
  title     = {Who Understands Whom? {M}utual Theory of Mind as a Unified Framework for Teacher-Facing {AI}},
  author    = {Baig, Duaa and Nurbakova, Diana and Calabretto, Sylvie and Mbaye, Baba},
  booktitle = {Technology Enhanced Learning (EC-TEL 2026)},
  publisher = {Springer},
  isbn      = {978-3-032-37981-8},
  url       = {https://link.springer.com/book/9783032379818},
  year      = {2026}
}
```

## License

This repository mixes two licenses:

| Scope | License |
|-------|---------|
| Source code (`src/`, `main.py`, `dtom-lens/` application code) | [MIT](LICENSE) |
| TalkMoves-derived files (`output/`, `reports/`, `dtom-lens/data/sample_transcripts/`, TalkMoves results in `dtom-lens/data/precomputed/`) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), inherited from TalkMoves |
| NCTE data | **Not redistributed.** Only aggregate statistics are included (`ncte_output/`, `dtom-lens/data/precomputed/ncte_results.json`) |

See [LICENSE-DATA.md](LICENSE-DATA.md) for the full file list and attribution requirements. To reproduce the NCTE results, apply for access to the NCTE dataset yourself (see [Datasets](#ncte)).

## References

- Suresh, A., Jacobs, J., Clevenger, C., Lai, V., Tan, C., Ward, W., Martin, J. H., & Sumner, T. (2022). The TalkMoves Dataset. *LREC 2022*.
- Berliner, D. C. (2004). Describing the behavior and documenting the accomplishments of expert teachers. *Bulletin of Science, Technology & Society*, 24(3), 200-212.
- Franke, M. L., Webb, N. M., Chan, A. G., Ing, M., Freund, D., & Battey, D. (2009). Teacher questioning to elicit students' mathematical thinking in elementary school classrooms. *Journal of Teacher Education*, 60(4), 380-392.
