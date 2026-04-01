# DToM — Double Theory of Mind Empirical Analysis

Empirical grounding for the Double Theory of Mind (DToM) framework through secondary analysis of classroom discourse data. This project accompanies a theoretical paper targeted at **EC-TEL 2026**.

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

## References

- Suresh, A., Jacobs, J., Clevenger, C., Lai, V., Tan, C., Ward, W., Martin, J. H., & Sumner, T. (2022). The TalkMoves Dataset. *LREC 2022*.
- Berliner, D. C. (2004). Describing the behavior and documenting the accomplishments of expert teachers. *Bulletin of Science, Technology & Society*, 24(3), 200-212.
- Franke, M. L., Webb, N. M., Chan, A. G., Ing, M., Freund, D., & Battey, D. (2009). Teacher questioning to elicit students' mathematical thinking in elementary school classrooms. *Journal of Teacher Education*, 60(4), 380-392.
