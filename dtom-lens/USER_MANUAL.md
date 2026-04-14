# DToM Lens — User Manual

**Version 1.0 — April 2026**

DToM Lens is an interactive web application that applies the Double Theory of Mind (DToM) framework to classroom discourse transcripts. It reveals hidden variation in teacher mentalizing depth that standard AI discourse classification systems collapse into single categories.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Interface Overview](#2-interface-overview)
3. [Sidebar](#3-sidebar)
4. [Tab 1: Transcript Explorer](#4-tab-1-transcript-explorer)
5. [Tab 2: Comparison View](#5-tab-2-comparison-view)
6. [Tab 3: Dashboard](#6-tab-3-dashboard)
7. [Tab 4: Within-Category Deep Dive](#7-tab-4-within-category-deep-dive)
8. [Tab 5: About](#8-tab-5-about)
9. [Uploading Your Own Data](#9-uploading-your-own-data)
10. [Understanding the Classifier](#10-understanding-the-classifier)
11. [Interpreting the Statistics](#11-interpreting-the-statistics)
12. [Deployment](#12-deployment)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Getting Started

### Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- A modern web browser (Chrome, Firefox, Safari, or Edge)

### Installation

```bash
cd dtom-lens
uv sync
```

This reads `pyproject.toml` and installs all dependencies into a local virtual environment.

To add a new dependency later:

```bash
uv add <package-name>
```

### Launching the Application

```bash
uv run streamlit run app.py
```

The application opens in your default browser at `http://localhost:8501`. If that port is busy, Streamlit will automatically select the next available port and display the URL in the terminal.

### Streamlit Cloud Deployment

To deploy on Streamlit Cloud:

1. Push the `dtom-lens/` directory to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repository.
3. Set the main file path to `app.py`.
4. Click **Deploy**.

No additional configuration is required — sample transcripts and precomputed results are bundled with the application.

---

## 2. Interface Overview

The application is organized into two areas:

- **Sidebar** (left): Data source selection, dataset sources, and framework overview.
- **Main area** (center): Five tabs, each providing a different view of the data.

```text
┌──────────────────┬──────────────────────────────────────────────────────┐
│  🔍 DToM Lens    │                                                      │
│  v1.0            │  [ Transcript Explorer | Comparison View |           │
│                  │    Dashboard | Within-Category | About ]             │
│  Data Source     │                                                      │
│  ○ Sample        │                                                      │
│  ○ Upload        │  (Active tab content)                                │
│  ○ Pre-computed  │                                                      │
│                  │                                                      │
│  [Transcript ▾]  │                                                      │
│                  │                                                      │
│  Datasets        │                                                      │
│  • TalkMoves     │                                                      │
│  • NCTE          │                                                      │
│                  │                                                      │
│  About DToM      │                                                      │
└──────────────────┴──────────────────────────────────────────────────────┘
```

---

## 3. Sidebar

The sidebar contains three sections:

### Data Source

Choose where transcripts come from:

- **Sample transcripts** (default) — Seven pre-loaded transcripts from TalkMoves, selected to span the full range of mentalizing depth:

  | Transcript | Characteristic |
  |---|---|
  | 7th grade math | Grade 7, shows student reasoning gradient |
  | Boats and Fish 1, Grade 4 | Near median L3 depth |
  | Boats and Fish 3, Grade 4 | High within-lesson variation (mean L3 = 0.50) |
  | Comparing Fractions 3, Grade 4 | High L3 depth (mean L3 = 0.68) |
  | Fraction as Number 1, Grade 4 | Lower L3 depth |
  | Gang of Four, Grade 4 | 75th percentile L3 depth |
  | Number line models 3, Grade 4 | 90th percentile L3 depth |

  Use the **Select transcript** dropdown to switch between them.

- **Upload CSV/XLSX** — Upload your own classroom transcript for analysis. See [Section 9](#9-uploading-your-own-data) for format requirements.

- **Pre-computed results** — When selected, the Transcript Explorer and Comparison View tabs display a message directing you to the Dashboard tab, which always shows pre-computed aggregate statistics from the full TalkMoves and NCTE datasets.

### Datasets

Source attribution for the two datasets used in the analysis:

- **TalkMoves** (Suresh et al., 2022) — 567 K-12 math transcripts. [GitHub link](https://github.com/SumnerLab/TalkMoves). Licensed CC BY-NC-SA 4.0.
- **NCTE** (Demszky et al., 2021) — 1,660 elementary math transcripts. [GitHub link](https://github.com/ddemszky/classroom-transcript-analysis). Restricted access — individual application required.

A caption reminds users that NCTE raw data is not displayed in the app; only aggregated statistics are shown per the data use agreement.

### About

- Brief description of the DToM framework.
- Expandable **"What is DToM?"** section explaining the three layers (L1 Behavioral, L2 Intermediate, L3 Deep).

---

## 4. Tab 1: Transcript Explorer

**Purpose:** Inspect a single transcript with color-coded mentalizing depth annotations on every teacher utterance.

The tab has a **fixed top section** (metrics, legend, filters) and a **scrollable transcript area** (600 px tall) — so statistics and filters remain visible while you browse the utterances.

### Fixed Top Section

**Summary metrics** (four cards): Total utterances, Teacher count, Student count, Mean L3 Depth.

**Color legend** — Three depth levels with color indicators:

| Color | Level | Meaning |
|---|---|---|
| Gray | A — Surface | Checking an answer or requesting a factual recall ("What is 3 times 5?") |
| Blue | B — Intermediate | Asking for procedural explanation ("How did you get that?") |
| Red | C — Deep | Probing conceptual understanding ("Why do you think that works?") |

**Filters** (three controls):

- **Filter by Talk Move** — Multiselect over the TalkMoves categories present in the transcript (PressAccuracy, PressReasoning, Restating, Revoicing, Marking, Context, KeepingTogether, GettingStudentsRelate). Empty selection shows all. You can pick multiple categories at once.
- **Filter by Depth** — Dropdown with options: All, A — Surface, B — Intermediate, C — Deep.
- **Show student utterances** — Checkbox toggling student turns on/off. Unchecking gives a teacher-only view.

Filters combine: selecting `PressAccuracy` + `C — Deep` shows only the deep questioning moves within that category.

### Scrollable Transcript Area

Each teacher utterance is rendered as an **expander** (click the caret to open). Inside the expander you see:

- The full utterance text, with a colored left border indicating its mentalizing depth.
- **Depth and Tag** line: `Depth: C (Deep - probing understanding) | Tag: PressReasoning`.
- **Preceding context**: the two turns that appeared just before this utterance.
- **Student response** that followed, with its tag and a checkmark (✓) if it was `ProvidingEvidence`.

Student utterances are shown inline with a light background and indented styling — not expanders — to keep the flow compact.

---

## 5. Tab 2: Comparison View

**Purpose:** The key demonstration of what standard AI discourse coding misses. The same transcript is displayed in two columns side by side.

The tab has a **fixed top section** (summary + filter + legend + column headers) and **two independently scrollable columns** (600 px tall each) — so headers and summary stay visible while you scroll through the transcript.

### Fixed Top Section

**Key insight callout** — A blue information box summarizing the whole transcript, e.g.:

> Standard AI coding uses **5 categories** for 162 teacher utterances. DToM reveals hidden depth variation in **3 of them** (covering 78 utterances, 48% of teacher talk).

**Per-category summary table** (expandable, shown by default) — Lists every TalkMoves category present in the transcript with:

| Column | Meaning |
|---|---|
| Category | The TalkMoves tag (e.g., PressAccuracy) |
| Total | Number of teacher utterances in this category |
| A (Surface) | Count at depth A |
| B (Intermediate) | Count at depth B |
| C (Deep) | Count at depth C |
| Depth levels | Number of distinct depth levels (1, 2, or 3) |

Categories with `Depth levels > 1` contain hidden variation that standard coding collapses.

**Category filter** — Multiselect letting you restrict the view to specific TalkMoves categories. By default all categories **except `None`** are selected (the `None` category contains procedural/management utterances with no talk-move label, and is usually noise for the comparison). Uncheck a category to hide it, or add `None` back in if you want to see it.

**Category color legend** — Colored chips showing the distinct color assigned to each TalkMoves category, e.g.:

- PressAccuracy (pink), PressReasoning (green), Restating (light blue), Revoicing (purple), Marking (teal), Context (amber), KeepingTogether (orange), GettingStudentsRelate (dark navy), None (light gray).

**Column headers** stay fixed above the two scrollable columns:

- **Standard AI View** (gray header) — TalkMoves labels only.
- **DToM Lens View** (purple header) — Mentalizing depth revealed.

### Scrollable Columns

Each column scrolls independently:

- **Left column (Standard AI View):** Each teacher utterance shows its TalkMoves tag with a **category-colored left border**, so you can visually see how the transcript is partitioned by standard categories.
- **Right column (DToM Lens View):** Each teacher utterance shows its **mentalizing depth (A/B/C) color-coded on the left border**, plus the original tag in small gray text for reference. Utterances that standard AI groups together (same category, left column) now reveal distinct depth levels (right column).

Student utterances appear identically in both columns (light background, tag label). The category filter hides student turns to keep category-focused views in context.

---

## 6. Tab 3: Dashboard

**Purpose:** Aggregate statistics across full datasets, presented as interactive charts. This tab is always populated, regardless of the data source selection.

### Dataset Selector

A dropdown at the top lets you choose:

- **TalkMoves** — 567 transcripts, 181,750 teacher utterances
- **NCTE** — 1,660 transcripts, 580,409 utterances
- **Both** — TalkMoves followed by NCTE replication

Just below the heading, a caption shows the dataset source citation and license.

### TalkMoves — Metric Cards

Five summary cards at the top:

| Card | TalkMoves Value | What It Means |
|---|---|---|
| Teacher Utterances | 181,750 | Total teacher turns analyzed |
| L1 Behavioral | 81.0% | Procedural/management — no modeling of student thinking |
| L2 Intermediate | 16.3% | Acknowledges student output without probing reasoning |
| L3 Deep | 2.7% | Actively probes student reasoning and mental models |
| Correlation (r) | 0.394 | Pearson correlation between transcript-level depth and student evidence rate |

### TalkMoves — Four Charts in One Row

To keep the dashboard compact, the four main charts are arranged in a single row. Each chart has a horizontal legend below the bars explaining the L1/L2/L3 colors, and any statistical annotation (χ², Cohen's d, r) appears just above the chart.

1. **Depth Distribution** — Utterance counts at each level (L1 Behavioral, L2 Intermediate, L3 Deep). Bars are labeled with percentages.

2. **Evidence by Depth** — Percentage of student responses providing evidence after each depth level. Demonstrates a steep gradient: 8.1% after L1 vs. 33.9% after L3 Deep. Annotation: `χ²=2485, p<.001`.

3. **Transcript Correlation** (scatter plot) — Each dot is one transcript. x-axis: mean depth, y-axis: % student evidence. Red trend line with annotation `r=0.394, p<.001, N=536`. Hover any dot to see the transcript name.

4. **Median Split** — Transcripts split at the median depth. Bars with error bars show high-depth transcripts have significantly higher student evidence rates. Annotation: `Cohen's d=0.47, p<.001`.

### Within-Category Analysis (Press for Accuracy)

A dedicated section below the four main charts shows Study 2 results for "Press for Accuracy":

**Four metric cards:** Total Utterances (20,770), Surface A (87.8%), Intermediate B (9.6%), Deep C (2.6%).

**Two charts in a half-width layout:**

- **Within-Category Distribution** — A/B/C percentages within PressAccuracy.
- **Evidence by Within-Cat Depth** — Student evidence rate by within-category depth (7.4% → 26.0% → 29.8%), with χ²=714, V=0.215 annotation.

### NCTE Replication

When NCTE or Both is selected, a separate section shows replication on the independent NCTE dataset.

A caption reminds users: **aggregated statistics only** — individual utterances and transcripts are not displayed, per the NCTE data use agreement. The caption links to the [NCTE GitHub repository](https://github.com/ddemszky/classroom-transcript-analysis) for raw data access applications.

### Interactivity

All charts support:

- **Hover** to see exact values and transcript names
- **Zoom** by dragging a selection box
- **Pan** by holding Shift and dragging
- **Reset** by double-clicking
- **Download** as PNG via the camera icon in the chart toolbar

---

## 7. Tab 4: Within-Category Deep Dive

**Purpose:** Explore mentalizing depth variation within any single talk move category.

### Category Selector

A dropdown lets you choose among all eight TalkMoves teacher categories:

- PressAccuracy, PressReasoning, Restating, Revoicing, Marking, Context, KeepingTogether, GettingStudentsRelate

### Pre-computed vs. Live Analysis

- **PressAccuracy** uses pre-computed results from the full TalkMoves dataset (20,770 utterances), providing the richest analysis.
- **All other categories** run the classifier live on whichever transcript is currently selected in the sidebar.

### Charts

Two charts side by side, each with a horizontal legend explaining A/B/C:

1. **Distribution** — Percentage of utterances at each depth level within the selected category. Bars are labeled with both the percentage and the count (e.g., `87.8% (n=18,232)`).

2. **Student Evidence Gradient** — Percentage of student responses providing evidence after each depth level. Annotation above the chart shows the chi-square result, Cramér's V, and the A→C difference ratio (e.g., `χ²=714, V=0.215 — 4.0× difference (A→C)`).

### Example Utterances

Below the charts, three columns show **up to five example utterances per depth level**, so the abstract A/B/C labels become concrete. Each column is headed by its colored depth label.

### LLM Convergent Validity

When viewing PressAccuracy, a section at the bottom shows agreement statistics between the rule-based classifier and an independent LLM-based classifier (Claude):

- **Cohen's κ** (0.229) — slight-to-fair agreement.
- **Agreement rate** (61%).
- **Model** used.

Key finding: the LLM identifies **more** non-surface depth than the rule-based classifier (47% vs. 13%), confirming that the rule-based estimates are conservative — real mentalizing depth variation is likely even larger than reported.

---

## 8. Tab 5: About

Provides background on the DToM framework, methodology, datasets, and citation information.

Contents:

- **The DToM Framework table** — L1 Behavioral, L2 Intermediate, L3 Deep with descriptions and examples.
- **Why It Matters** — explains what standard tools miss and quotes the 87.8% / 9.6% / 2.6% finding within PressAccuracy.
- **Methodology** — dataset summaries (TalkMoves and NCTE with links), classifier description, statistical approach.
- **Citation** — suggested format.
- **Links** — paper and source code (availability noted).

---

## 9. Uploading Your Own Data

### File Format

The application accepts `.csv` and `.xlsx` files. Your file must contain at minimum:

| Column | Required | Description |
|---|---|---|
| `Sentence` or `text` | Yes | The utterance text |
| `Speaker` | Yes | Who spoke (e.g., "Teacher", "Student", a name) |
| `Teacher Tag` or `teacher_tag` | No | Standard talk move category (e.g., "Press for Accuracy") |
| `Student Tag` or `student_tag` | No | Student response category (e.g., "Providing Evidence") |

Column names are case-insensitive. The application automatically maps common variants.

### What Happens on Upload

1. The file is validated for required columns.
2. Tags are normalized to canonical categories (if present).
3. L3 depth is assigned based on the teacher tag.
4. The within-category mentalizing depth classifier runs on all teacher utterances.
5. Tabs 1, 2, and 4 update to display the uploaded transcript.

### Without Tags

If your file has only `Sentence` and `Speaker` columns (no talk move tags), the application will still run the within-category mentalizing depth classifier on all utterances. However, the L3 depth mapping and Comparison View will not be available since they depend on standard talk move labels.

### File Size

The maximum upload size is 50 MB (configurable in `.streamlit/config.toml`).

---

## 10. Understanding the Classifier

DToM Lens uses a **rule-based classifier** that operates at two levels:

### Level 1: Talk Move → Depth Layer

Each standard TalkMoves category is mapped to a DToM framework layer (L1/L2/L3) based on the cognitive demands it places on teacher mentalizing:

| Layer | Categories | Teacher Cognition |
|---|---|---|
| L1 Behavioral | None, Context, KeepingTogether, Marking | Procedural/management — no need to model student thinking |
| L2 Intermediate | Restating, Revoicing, PressAccuracy | Acknowledges student output but does not probe reasoning |
| L3 Deep | PressReasoning, GettingStudentsRelate | Actively models and probes student mental states |

### Level 2: Within-Category Mentalizing Depth (Linguistic Markers → A/B/C)

Within any single talk move category, the classifier uses linguistic pattern matching to distinguish three depth levels:

**Level C (Deep)** — detected by patterns such as:

- "why do you think...", "how do you know...", "what makes you..."
- "prove", "justify", "convince"
- "what if...", "what would happen..."
- "does that make sense", "is there another way"

**Level B (Intermediate)** — detected by patterns such as:

- "how did you...", "what did you do..."
- "show me/us", "explain", "describe"
- "tell me/us how/what..."
- "how did you figure/solve/find..."

**Level A (Surface)** — the default when neither deep nor intermediate patterns match. These are typically factual recall or answer-checking questions.

The classifier applies **priority ordering**: Deep > Intermediate > Surface. An utterance matching both deep and intermediate patterns is classified as deep.

---

## 11. Interpreting the Statistics

### Chi-Square Test (χ²)

Tests whether the relationship between teacher mentalizing depth and student response quality is statistically significant (i.e., not due to chance). A large χ² with p < .001 indicates a highly significant association.

### Cramer's V

An effect size measure for the chi-square test, ranging from 0 (no association) to 1 (perfect association). Values in this analysis range from 0.089 to 0.215, indicating small-to-medium effects — meaningful given the naturalistic classroom data.

### Pearson r

A correlation coefficient measuring the linear relationship between transcript-level mentalizing depth and student evidence rates. The value r = 0.394 indicates a moderate positive correlation: transcripts with higher average depth tend to have higher rates of student evidence-providing.

### Cohen's d

An effect size for the median-split comparison. The value d = 0.468 is a medium effect, meaning that high-depth transcripts have meaningfully higher student reasoning than low-depth transcripts.

### Cohen's Kappa (κ)

A measure of inter-rater agreement between the rule-based and LLM classifiers, correcting for chance agreement. The value κ = 0.229 indicates slight-to-fair agreement — expected, since the LLM uses entirely different reasoning to arrive at depth judgments.

---

## 12. Deployment

### Local

```bash
uv run streamlit run app.py
```

### Streamlit Cloud

1. Ensure `requirements.txt`, `app.py`, and the `data/` directory are in the repository root (or adjust paths).
2. Connect the repository at [share.streamlit.io](https://share.streamlit.io).
3. No environment variables or secrets are required for basic operation.

### Configuration

The file `.streamlit/config.toml` controls the visual theme and server settings:

```toml
[theme]
primaryColor = "#534AB7"        # Purple accent
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F7FC"
textColor = "#2C2C2A"
font = "sans serif"

[server]
maxUploadSize = 50              # MB
```

---

## 13. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` on startup | Run `uv sync` to install dependencies |
| Plotly/Streamlit version conflict (`AttributeError: ... Icicle`) | Upgrade both: `uv add plotly@latest streamlit@latest` |
| `VIRTUAL_ENV does not match project environment` warning | Harmless — caused by the parent project's venv being active. Either `deactivate` first or ignore. |
| Port 8501 already in use | Streamlit auto-selects the next port, or specify one: `uv run streamlit run app.py --server.port 8502` |
| Sample transcripts not loading | Ensure `.xlsx` files are present in `data/sample_transcripts/` |
| Upload fails with column error | Check that your file has `Sentence` (or `text`) and `Speaker` columns |
| Charts appear blank | Check browser console for JavaScript errors; try a different browser or clear cache |
| Slow startup | First load processes sample transcripts (~2 seconds). Subsequent loads use Streamlit's cache. |
