"""
DToM Lens — Interactive Diagnostic Tool for Mentalizing Depth in Teacher Discourse
===================================================================================
Streamlit application for the L@S 2026 demo paper.

Usage:
    streamlit run app.py
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dtom_classifier import (
    CATEGORY_COLORS,
    DEPTH_COLORS,
    L3_COLORS,
    L3_DEPTH_MAP,
    L3_LABELS,
    MENTAL_DEPTH_LABELS,
    classify_mentalizing_depth,
    normalize_student_tag,
    normalize_teacher_tag,
    process_transcript,
)
from dtom_analysis import (
    analyze_sequential,
    analyze_transcript_level,
    analyze_within_category,
    get_transcript_summary,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DToM Lens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .utterance-box {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .utterance-teacher {
        border-left: 4px solid;
        background-color: #fafafa;
    }
    .utterance-student {
        background-color: #f0f4f8;
        border-left: 4px solid #ddd;
        margin-left: 20px;
    }
    .depth-label {
        color: gray;
        font-size: 0.8em;
    }
    .metric-highlight {
        font-size: 1.5em;
        font-weight: bold;
        color: #534AB7;
    }
    .comparison-header {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    div[data-testid="stMetric"] {
        background-color: #F8F7FC;
        padding: 12px;
        border-radius: 8px;
    }
    /* Larger tab labels */
    button[data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PRECOMPUTED_DIR = os.path.join(DATA_DIR, 'precomputed')
SAMPLE_DIR = os.path.join(DATA_DIR, 'sample_transcripts')


@st.cache_data
def load_precomputed():
    """Load all precomputed analysis results."""
    results = {}
    for name in ['talkmoves_results', 'ncte_results', 'llm_agreement']:
        path = os.path.join(PRECOMPUTED_DIR, f'{name}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[name] = json.load(f)
    # Load transcript-level CSV
    csv_path = os.path.join(PRECOMPUTED_DIR, 'transcript_l3_analysis.csv')
    if os.path.exists(csv_path):
        results['transcript_data'] = pd.read_csv(csv_path)
    return results


@st.cache_data
def load_sample_transcripts():
    """Load and process all sample transcripts."""
    transcripts = {}
    files = sorted(glob.glob(os.path.join(SAMPLE_DIR, '*.xlsx')))
    for f in files:
        name = Path(f).stem
        try:
            df = pd.read_excel(f, engine='openpyxl')
            df['transcript_id'] = name
            df = process_transcript(df)
            transcripts[name] = df
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    return transcripts


def process_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    """Process an uploaded CSV or XLSX file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        df['transcript_id'] = Path(uploaded_file.name).stem
        df = process_transcript(df)
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(precomputed, sample_transcripts):
    """Render the sidebar with data source and classifier options."""
    with st.sidebar:
        st.title("🔍 DToM Lens")
        st.caption("v1.0")

        st.markdown("---")
        st.subheader("Data Source")

        data_source = st.radio(
            "Choose data source",
            ["Sample transcripts", "Upload CSV/XLSX", "Pre-computed results"],
            label_visibility="collapsed",
        )

        active_transcript = None
        active_df = None

        if data_source == "Sample transcripts":
            transcript_names = list(sample_transcripts.keys())
            if transcript_names:
                selected = st.selectbox("Select transcript", transcript_names)
                active_transcript = selected
                active_df = sample_transcripts.get(selected)
            else:
                st.warning("No sample transcripts found.")

        elif data_source == "Upload CSV/XLSX":
            uploaded = st.file_uploader(
                "Upload a transcript file",
                type=['csv', 'xlsx'],
                help="File must have 'Sentence' and 'Speaker' columns. "
                     "Optionally include 'Teacher Tag' and 'Student Tag'.",
            )
            if uploaded:
                active_df = process_uploaded_file(uploaded)
                if active_df is not None:
                    active_transcript = Path(uploaded.name).stem

        st.markdown("---")
        st.subheader("Datasets")
        st.markdown(
            "**TalkMoves** (Suresh et al., 2022)  \n"
            "567 K-12 math transcripts.  \n"
            "[GitHub](https://github.com/SumnerLab/TalkMoves) "
            "| CC BY-NC-SA 4.0"
        )
        st.markdown(
            "**NCTE** (Demszky et al., 2021)  \n"
            "1,660 elementary math transcripts.  \n"
            "[GitHub](https://github.com/ddemszky/classroom-transcript-analysis) "
            "| Restricted access"
        )
        st.caption(
            "NCTE data requires individual access application. "
            "Only aggregated statistics are shown here."
        )

        st.markdown("---")
        st.subheader("About")
        st.markdown(
            "**Framework:** Double Theory of Mind (DToM)  \n"
            "Standard AI discourse classification collapses "
            "meaningful variation in teacher mentalizing depth."
        )
        with st.expander("What is DToM?"):
            st.markdown(
                "The DToM framework identifies three layers of "
                "theory-of-mind in educational AI systems:\n\n"
                "- **L1 — Behavioral:** Recognizing observable behavior\n"
                "- **L2 — Intermediate:** Modeling underlying processes\n"
                "- **L3 — Deep:** Understanding *why* — the conceptual "
                "reasoning and mental models behind behavior\n\n"
                "Current AI discourse tools primarily operate at L1/L2, "
                "collapsing meaningful L3 variation."
            )

    return data_source, active_transcript, active_df


# ============================================================
# TAB 1: TRANSCRIPT EXPLORER
# ============================================================

def render_transcript_explorer(active_transcript, active_df):
    """Render the Transcript Explorer tab."""
    if active_df is None:
        st.info("Select a transcript from the sidebar to explore.")
        return

    st.subheader(f"Transcript: {active_transcript}")

    # Summary metrics
    summary = get_transcript_summary(active_df)
    cols = st.columns(4)
    cols[0].metric("Utterances", summary['total_utterances'])
    cols[1].metric("Teacher", summary['teacher_utterances'])
    cols[2].metric("Student", summary['student_utterances'])
    if summary['mean_l3_depth'] is not None:
        cols[3].metric("Mean L3 Depth", f"{summary['mean_l3_depth']:.3f}")

    st.markdown("---")

    # Color legend
    legend_cols = st.columns(3)
    legend_cols[0].markdown(
        '<span style="color: #95a5a6; font-weight: bold;">■</span> '
        'Level A — Surface (checking answer/fact)',
        unsafe_allow_html=True,
    )
    legend_cols[1].markdown(
        '<span style="color: #3498db; font-weight: bold;">■</span> '
        'Level B — Intermediate (requesting procedure)',
        unsafe_allow_html=True,
    )
    legend_cols[2].markdown(
        '<span style="color: #e74c3c; font-weight: bold;">■</span> '
        'Level C — Deep (probing understanding)',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
    with filter_col1:
        available_tags = sorted(
            [t for t in active_df['t_move'].dropna().unique() if t != 'Other']
        )
        tag_filter = st.multiselect(
            "Filter by Talk Move (empty = all)",
            available_tags,
            default=[],
            key="explorer_tag_filter",
        )
    with filter_col2:
        depth_filter = st.selectbox(
            "Filter by Depth",
            ['All', 'A — Surface', 'B — Intermediate', 'C — Deep'],
            key="explorer_depth_filter",
        )
    with filter_col3:
        show_students = st.checkbox("Show student utterances", value=True, key="explorer_show_students")

    depth_filter_val = depth_filter[0] if depth_filter != 'All' else None

    st.markdown("---")

    # Scrollable transcript area (fixed height keeps header/filters visible)
    transcript_container = st.container(height=600)

    with transcript_container:
        for i, (idx, row) in enumerate(active_df.iterrows()):
            text = str(row.get('Sentence', '')).strip()
            if not text or text == 'nan':
                continue

            speaker = str(row.get('Speaker', '')).strip()
            is_teacher = row.get('t_move') is not None and pd.notna(row.get('t_move'))

            # Apply filters
            if is_teacher:
                if tag_filter and row.get('t_move') not in tag_filter:
                    continue
                if depth_filter_val and row.get('mental_depth') != depth_filter_val:
                    continue
            else:
                if not show_students:
                    continue

            if is_teacher:
                depth = row.get('mental_depth', 'A')
                color = DEPTH_COLORS.get(depth, '#95a5a6')
                tag = row.get('t_move', '')
                depth_label = MENTAL_DEPTH_LABELS.get(depth, '')

                with st.expander(
                    f"Turn {i+1} [T] — {text[:80]}{'...' if len(text) > 80 else ''}",
                    expanded=False,
                ):
                    st.markdown(
                        f'<div class="utterance-box utterance-teacher" '
                        f'style="border-left-color: {color};">'
                        f'<b>[{speaker}]</b> {text}<br>'
                        f'<span class="depth-label">Depth: {depth} ({depth_label}) '
                        f'| Tag: {tag}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Show context: preceding turns
                    if i >= 2:
                        st.caption("Preceding context:")
                        for j in range(max(0, i - 2), i):
                            prev_row = active_df.iloc[j]
                            prev_text = str(prev_row.get('Sentence', '')).strip()
                            prev_speaker = str(prev_row.get('Speaker', '')).strip()
                            if prev_text and prev_text != 'nan':
                                st.caption(f"  [{prev_speaker}] {prev_text}")

                    # Show next student response
                    for offset in range(1, 4):
                        if i + offset < len(active_df):
                            next_row = active_df.iloc[i + offset]
                            if pd.notna(next_row.get('s_move')):
                                next_text = str(next_row.get('Sentence', '')).strip()
                                next_tag = next_row.get('s_move', '')
                                is_evidence = next_tag == 'ProvidingEvidence'
                                st.caption(
                                    f"→ Student response: \"{next_text}\" "
                                    f"({next_tag}{'  ✓' if is_evidence else ''})"
                                )
                                break
            else:
                # Student utterance — compact display
                s_tag = row.get('s_move', '')
                is_evidence = s_tag == 'ProvidingEvidence'
                st.markdown(
                    f'<div class="utterance-box utterance-student">'
                    f'<b>[{speaker}]</b> {text}<br>'
                    f'<span class="depth-label">Tag: {s_tag}'
                    f'{"  ✓" if is_evidence else ""}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# TAB 2: COMPARISON VIEW
# ============================================================

def render_comparison_view(active_transcript, active_df):
    """Render the Comparison View tab — side-by-side standard vs DToM."""
    if active_df is None:
        st.info("Select a transcript from the sidebar to compare views.")
        return

    st.subheader(f"Comparison: {active_transcript}")
    st.markdown(
        "The same transcript viewed through **standard AI coding** (left) "
        "vs. **DToM Lens** (right). Notice how standard coding assigns a "
        "single label while DToM reveals distinct mentalizing depths."
    )

    # Summary insight
    teacher_df = active_df[active_df['t_move'].notna()]
    if len(teacher_df) == 0:
        st.warning("No teacher utterances with tags found in this transcript.")
        return

    # Build a per-category summary table
    summary_rows = []
    if 'mental_depth' in teacher_df.columns:
        for tag, cat in teacher_df.groupby('t_move'):
            depths = cat['mental_depth'].value_counts()
            n_a = int(depths.get('A', 0))
            n_b = int(depths.get('B', 0))
            n_c = int(depths.get('C', 0))
            total = len(cat)
            summary_rows.append({
                'Category': tag,
                'Total': total,
                'A (Surface)': n_a,
                'B (Intermediate)': n_b,
                'C (Deep)': n_c,
                'Depth levels': depths.nunique(),
            })
    summary_df = pd.DataFrame(summary_rows).sort_values('Total', ascending=False) if summary_rows else pd.DataFrame()

    # Key insight: identify categories that have hidden variation (>1 depth level)
    varied = summary_df[summary_df['Depth levels'] > 1] if not summary_df.empty else pd.DataFrame()
    n_varied = len(varied)
    total_teacher = len(teacher_df)

    if n_varied > 0:
        varied_utts = int(varied['Total'].sum())
        st.info(
            f"**Key insight:** Standard AI coding uses **{len(summary_df)} categories** "
            f"for {total_teacher} teacher utterances. DToM reveals hidden depth variation in "
            f"**{n_varied} of them** (covering {varied_utts} utterances, "
            f"{varied_utts/total_teacher*100:.0f}% of teacher talk)."
        )
    else:
        st.info(
            f"**Key insight:** Standard AI coding uses **{len(summary_df)} categories** "
            f"for {total_teacher} teacher utterances. "
            f"In this particular transcript, each category has a single depth level — "
            f"hidden variation may be more visible in other transcripts."
        )

    # Summary table at top
    if not summary_df.empty:
        with st.expander("Per-category summary (click to expand)", expanded=True):
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Category filter for the views
    all_categories = summary_df['Category'].tolist() if not summary_df.empty else []
    selected_categories = st.multiselect(
        "Filter categories (empty = show all)",
        all_categories,
        default=[],
        key="comparison_cat_filter",
    )

    # Color legend for the categories shown
    legend_tags = selected_categories if selected_categories else all_categories
    if legend_tags:
        legend_html = '<div style="margin: 8px 0; font-size: 0.85em;">'
        legend_html += '<b>Category colors:</b> '
        for t in legend_tags:
            c = CATEGORY_COLORS.get(t, '#7f8c8d')
            legend_html += (
                f'<span style="background-color: {c}; color: white; '
                f'padding: 2px 8px; border-radius: 3px; margin-right: 4px; '
                f'display: inline-block; margin-bottom: 2px;">{t}</span>'
            )
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)

    # Fixed column headers
    header_col1, header_col2 = st.columns(2)

    with header_col1:
        st.markdown(
            '<div class="comparison-header" style="background-color: #e8e8e8;">'
            '<h4 style="margin:0;">Standard AI View</h4>'
            '<p style="color: gray; margin:0;">TalkMoves labels only</p></div>',
            unsafe_allow_html=True,
        )

    with header_col2:
        st.markdown(
            '<div class="comparison-header" style="background-color: #F8F7FC;">'
            '<h4 style="margin:0;">DToM Lens View</h4>'
            '<p style="color: gray; margin:0;">Mentalizing depth revealed</p></div>',
            unsafe_allow_html=True,
        )

    # Scrollable body: two containers of fixed height, one per column
    body_col1, body_col2 = st.columns(2)
    left_scroll = body_col1.container(height=600)
    right_scroll = body_col2.container(height=600)

    # Display utterances side by side — each column scrolls independently
    for i, (idx, row) in enumerate(active_df.iterrows()):
        text = str(row.get('Sentence', '')).strip()
        if not text or text == 'nan':
            continue

        speaker = str(row.get('Speaker', '')).strip()
        is_teacher = row.get('t_move') is not None and pd.notna(row.get('t_move'))

        if is_teacher:
            tag = row.get('t_move', '')
            # Apply category filter (only for teacher turns)
            if selected_categories and tag not in selected_categories:
                continue

            depth = row.get('mental_depth', 'A')
            depth_color = DEPTH_COLORS.get(depth, '#95a5a6')
            cat_color = CATEGORY_COLORS.get(tag, '#7f8c8d')
            depth_label = MENTAL_DEPTH_LABELS.get(depth, '')

            with left_scroll:
                # Standard AI View — colored by category
                st.markdown(
                    f'<div class="utterance-box" style="border-left: 4px solid {cat_color}; '
                    f'background-color: #fafafa; padding: 8px 12px;">'
                    f'<b>[T]</b> <span style="color: {cat_color}; font-weight: bold;">'
                    f'{tag}</span><br>'
                    f'<small style="color: #666;">{text[:100]}'
                    f'{"..." if len(text) > 100 else ""}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with right_scroll:
                # DToM Lens View — colored by depth
                st.markdown(
                    f'<div class="utterance-box" style="border-left: 4px solid {depth_color}; '
                    f'background-color: #fafafa; padding: 8px 12px;">'
                    f'<b>[T]</b> <span style="color: {depth_color}; font-weight: bold;">'
                    f'{depth} — {depth_label}</span>'
                    f'<span style="color: gray; font-size: 0.85em;"> · {tag}</span><br>'
                    f'<small style="color: #666;">{text[:100]}'
                    f'{"..." if len(text) > 100 else ""}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            # Student utterances: only show if no category filter is active
            # (since filter is teacher-only, showing student turns would be out of context)
            if selected_categories:
                continue
            s_tag = row.get('s_move', '')
            for scroll_container in [left_scroll, right_scroll]:
                with scroll_container:
                    st.markdown(
                        f'<div class="utterance-box utterance-student">'
                        f'<b>[S]</b> {text[:100]}'
                        f'{"..." if len(text) > 100 else ""}<br>'
                        f'<span class="depth-label">{s_tag}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ============================================================
# TAB 3: DASHBOARD
# ============================================================

def _make_level_bar_chart(
    values: list,
    labels: list,
    legend_names: list,
    colors: list,
    y_title: str,
    text_values: list | None = None,
    y_range: tuple | None = None,
    annotation: str | None = None,
) -> go.Figure:
    """Build a bar chart with one trace per level so legends render properly."""
    fig = go.Figure()
    for val, lbl, name, color, txt in zip(
        values, labels, legend_names, colors,
        text_values if text_values else [f"{v}" for v in values],
    ):
        fig.add_trace(go.Bar(
            x=[lbl],
            y=[val],
            name=name,
            marker_color=color,
            text=[txt],
            textposition='outside',
            showlegend=True,
        ))

    # Give headroom for outside text labels
    y_max = max(values) if values else 1
    if y_range is None:
        y_range = (0, y_max * 1.25) if y_max > 0 else (0, 1)

    fig.update_layout(
        height=260,
        margin=dict(t=20, b=20, l=10, r=10),
        yaxis_title=y_title,
        yaxis_range=list(y_range),
        font=dict(size=10),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.3,
            xanchor="center", x=0.5,
            font=dict(size=9),
        ),
        barmode='group',
    )

    if annotation:
        fig.add_annotation(
            text=annotation,
            xref="paper", yref="paper",
            x=0.5, y=1.02, showarrow=False,
            xanchor='center',
            font=dict(size=10, color="gray"),
        )

    return fig




def render_dashboard(precomputed):
    """Render the Dashboard tab with aggregate statistics."""
    st.subheader("Aggregate Statistics")

    # Dataset selector
    available_datasets = []
    if 'talkmoves_results' in precomputed:
        available_datasets.append('TalkMoves')
    if 'ncte_results' in precomputed:
        available_datasets.append('NCTE')
    if len(available_datasets) > 1:
        available_datasets.append('Both')

    if not available_datasets:
        st.warning("No precomputed results found.")
        return

    dataset = st.selectbox("Dataset", available_datasets)

    if dataset == 'TalkMoves' or dataset == 'Both':
        _render_talkmoves_dashboard(precomputed)

    if dataset == 'NCTE':
        _render_ncte_dashboard(precomputed)

    if dataset == 'Both':
        st.markdown("---")
        st.subheader("NCTE Replication")
        _render_ncte_dashboard(precomputed)


def _render_talkmoves_dashboard(precomputed):
    """Render TalkMoves dataset dashboard."""
    tm = precomputed.get('talkmoves_results', {})
    s1 = tm.get('study1_l3_mapping', {})
    s2 = tm.get('study2_within_category', {})

    # Summary metric cards
    st.markdown("#### TalkMoves Dataset")
    st.caption(
        "Source: [Suresh et al. (2022)](https://github.com/SumnerLab/TalkMoves) "
        "| 567 K-12 math transcripts | CC BY-NC-SA 4.0"
    )
    dist = s1.get('distribution', {})
    cols = st.columns(5)
    cols[0].metric(
        "Teacher Utterances",
        f"{s1.get('total_teacher_utterances', 0):,}",
    )
    cols[1].metric(
        "L1 Behavioral",
        f"{dist.get('level_0', {}).get('pct', 0)}%",
        help="Procedural/management — no modeling of student thinking",
    )
    cols[2].metric(
        "L2 Intermediate",
        f"{dist.get('level_1', {}).get('pct', 0)}%",
        help="Acknowledge/check student output without probing reasoning",
    )
    cols[3].metric(
        "L3 Deep",
        f"{dist.get('level_2', {}).get('pct', 0)}%",
        help="Actively probes student reasoning and mental models",
    )
    cols[4].metric(
        "Correlation (r)",
        f"{s1.get('transcript_level', {}).get('pearson_r', 0):.3f}",
        help="Pearson r: mean L3 depth vs student evidence rate (p < .001)",
    )

    # All four L3 charts in one row
    chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)

    l3_labels = ['L1', 'L2', 'L3']
    l3_legend = ['L1 Behavioral', 'L2 Intermediate', 'L3 Deep']
    l3_colors = ['#95a5a6', '#3498db', '#e74c3c']

    with chart_col1:
        st.markdown("##### Depth Distribution")
        dist = s1.get('distribution', {})
        counts = [
            dist.get('level_0', {}).get('count', 0),
            dist.get('level_1', {}).get('count', 0),
            dist.get('level_2', {}).get('count', 0),
        ]
        pcts = [
            dist.get('level_0', {}).get('pct', 0),
            dist.get('level_1', {}).get('pct', 0),
            dist.get('level_2', {}).get('pct', 0),
        ]
        fig = _make_level_bar_chart(
            values=counts,
            labels=l3_labels,
            legend_names=l3_legend,
            colors=l3_colors,
            y_title="Utterances",
            text_values=[f"{p}%" for p in pcts],
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown("##### Evidence by Depth")
        seq = s1.get('sequential', {}).get('by_depth', {})
        vals = [
            seq.get('level_0', {}).get('evidence_pct', 0),
            seq.get('level_1', {}).get('evidence_pct', 0),
            seq.get('level_2', {}).get('evidence_pct', 0),
        ]
        chi2 = s1.get('sequential', {}).get('chi2', 0)
        fig = _make_level_bar_chart(
            values=vals,
            labels=l3_labels,
            legend_names=l3_legend,
            colors=l3_colors,
            y_title="% Student Evidence",
            text_values=[f"{v}%" for v in vals],
            annotation=f"χ²={chi2}, p<.001",
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col3:
        st.markdown("##### Transcript Correlation")
        transcript_data = precomputed.get('transcript_data')
        if transcript_data is not None and len(transcript_data) > 0:
            fig = px.scatter(
                transcript_data,
                x='mean_l3',
                y='pct_evidence',
                hover_data=['transcript_id'],
                labels={
                    'mean_l3': 'Mean L3 Depth per Transcript',
                    'pct_evidence': '% Student Evidence',
                },
                opacity=0.5,
            )
            fig.update_traces(marker=dict(color='#2c3e50', size=6))
            # Add trend line
            z = np.polyfit(transcript_data['mean_l3'], transcript_data['pct_evidence'], 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(
                transcript_data['mean_l3'].min(),
                transcript_data['mean_l3'].max(),
                100,
            )
            fig.add_trace(go.Scatter(
                x=x_range, y=p_line(x_range),
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
            ))
            r = s1.get('transcript_level', {}).get('pearson_r', 0)
            n = s1.get('transcript_level', {}).get('n', 0)
            fig.update_layout(
                height=260,
                margin=dict(t=30, b=20, l=10, r=10),
                font=dict(size=10),
                xaxis_title="Mean Depth",
                yaxis_title="% Student Evidence",
            )
            fig.add_annotation(
                text=f"r={r:.3f}, p<.001, N={n}",
                xref="paper", yref="paper",
                x=0.5, y=1.02, showarrow=False,
                xanchor='center',
                font=dict(size=10, color="gray"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Transcript-level data not available.")

    with chart_col4:
        st.markdown("##### Median Split")
        ms = s1.get('median_split', {})
        low_val = ms.get('low_mean', 0) * 100
        high_val = ms.get('high_mean', 0) * 100
        low_sd = ms.get('low_sd', 0) * 100
        high_sd = ms.get('high_sd', 0) * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Low'], y=[low_val],
            name='Low Depth',
            marker_color='#3498db',
            text=[f"{low_val:.1f}%"], textposition='outside',
            error_y=dict(type='data', array=[low_sd], visible=True),
        ))
        fig.add_trace(go.Bar(
            x=['High'], y=[high_val],
            name='High Depth',
            marker_color='#e74c3c',
            text=[f"{high_val:.1f}%"], textposition='outside',
            error_y=dict(type='data', array=[high_sd], visible=True),
        ))

        d = ms.get('cohens_d', 0)
        y_max = max(low_val + low_sd, high_val + high_sd)
        fig.update_layout(
            height=260, margin=dict(t=30, b=20, l=10, r=10),
            yaxis_title="% Student Evidence",
            yaxis_range=[0, y_max * 1.3],
            font=dict(size=10),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.3,
                xanchor="center", x=0.5,
                font=dict(size=9),
            ),
        )
        fig.add_annotation(
            text=f"Cohen's d={d:.2f}, p<.001",
            xref="paper", yref="paper",
            x=0.5, y=1.02, showarrow=False,
            xanchor='center',
            font=dict(size=10, color="gray"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Within-category summary (Study 2)
    st.markdown("---")
    st.markdown("#### Within-Category Analysis (Press for Accuracy)")

    wc_cols = st.columns(4)
    wc_cols[0].metric("Total Utterances", f"{s2.get('total_utterances', 0):,}")
    wc_cols[1].metric(
        "Surface (A)",
        f"{s2.get('distribution', {}).get('level_A', {}).get('pct', 0)}%",
    )
    wc_cols[2].metric(
        "Intermediate (B)",
        f"{s2.get('distribution', {}).get('level_B', {}).get('pct', 0)}%",
    )
    wc_cols[3].metric(
        "Deep (C)",
        f"{s2.get('distribution', {}).get('level_C', {}).get('pct', 0)}%",
    )

    wc_chart1, wc_chart2, _, _ = st.columns(4)
    wc_labels = ['A', 'B', 'C']
    wc_legend = ['A — Surface', 'B — Intermediate', 'C — Deep']
    wc_colors = ['#95a5a6', '#3498db', '#e74c3c']

    with wc_chart1:
        st.markdown("##### Within-Category Distribution")
        wc_dist = s2.get('distribution', {})
        pcts = [
            wc_dist.get('level_A', {}).get('pct', 0),
            wc_dist.get('level_B', {}).get('pct', 0),
            wc_dist.get('level_C', {}).get('pct', 0),
        ]
        fig = _make_level_bar_chart(
            values=pcts,
            labels=wc_labels,
            legend_names=wc_legend,
            colors=wc_colors,
            y_title="% of Utterances",
            text_values=[f"{p}%" for p in pcts],
            y_range=(0, 100),
        )
        st.plotly_chart(fig, use_container_width=True)

    with wc_chart2:
        st.markdown("##### Evidence by Within-Cat Depth")
        val = s2.get('validation', {}).get('by_depth', {})
        vals = [
            val.get('A', {}).get('evidence_pct', 0),
            val.get('B', {}).get('evidence_pct', 0),
            val.get('C', {}).get('evidence_pct', 0),
        ]
        chi2 = s2.get('validation', {}).get('chi2', 0)
        v = s2.get('validation', {}).get('cramers_v', 0)
        fig = _make_level_bar_chart(
            values=vals,
            labels=wc_labels,
            legend_names=wc_legend,
            colors=wc_colors,
            y_title="% Student Evidence",
            text_values=[f"{v_}%" for v_ in vals],
            annotation=f"χ²={chi2}, p<.001, V={v}",
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_ncte_dashboard(precomputed):
    """Render NCTE replication dashboard."""
    ncte = precomputed.get('ncte_results', {})
    if not ncte:
        st.warning("NCTE results not available.")
        return

    st.markdown("#### NCTE Dataset")
    st.caption(
        "Aggregated statistics only — individual transcripts are not displayed "
        "per NCTE data use agreement. "
        "To access the raw data, apply via "
        "[GitHub](https://github.com/ddemszky/classroom-transcript-analysis)."
    )

    ds = ncte.get('dataset', {})
    r1 = ncte.get('study_r1', {})
    r2 = ncte.get('study_r2', {})

    cols = st.columns(4)
    cols[0].metric("Utterances", f"{ds.get('total_utterances', 0):,}")
    cols[1].metric("Transcripts", f"{ds.get('n_transcripts', 0):,}")
    cols[2].metric(
        "Surface (A)",
        f"{r1.get('distribution', {}).get('A', {}).get('pct', 0)}%",
    )
    cols[3].metric(
        "Correlation (r)",
        f"{r2.get('transcript_level', {}).get('pearson_r_elaborate', 0):.3f}",
    )

    ncte_c1, ncte_c2, _, _ = st.columns(4)

    ncte_labels = ['A', 'B', 'C']
    ncte_legend = ['A — Surface', 'B — Intermediate', 'C — Deep']
    ncte_colors = ['#95a5a6', '#3498db', '#e74c3c']

    with ncte_c1:
        st.markdown("##### Depth Distribution (NCTE)")
        dist = r1.get('distribution', {})
        pcts = [
            dist.get('A', {}).get('pct', 0),
            dist.get('B', {}).get('pct', 0),
            dist.get('C', {}).get('pct', 0),
        ]
        fig = _make_level_bar_chart(
            values=pcts,
            labels=ncte_labels,
            legend_names=ncte_legend,
            colors=ncte_colors,
            y_title="% Teacher Utterances",
            text_values=[f"{p}%" for p in pcts],
            y_range=(0, 100),
        )
        st.plotly_chart(fig, use_container_width=True)

    with ncte_c2:
        st.markdown("##### Student Elaboration (NCTE)")
        seq = r2.get('sequential', {}).get('by_depth', {})
        vals = [
            seq.get('A', {}).get('elaborate_pct', 0),
            seq.get('B', {}).get('elaborate_pct', 0),
            seq.get('C', {}).get('elaborate_pct', 0),
        ]
        chi2 = r2.get('sequential', {}).get('chi2', 0)
        fig = _make_level_bar_chart(
            values=vals,
            labels=ncte_labels,
            legend_names=ncte_legend,
            colors=ncte_colors,
            y_title="% Elaborated Response",
            text_values=[f"{v_}%" for v_ in vals],
            annotation=f"χ²={chi2:.1f}, p<.001",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Replication verdict — R1 (Distribution): {r1.get('replication_verdict', 'N/A')} | "
        f"R2 (Sequential): {r2.get('replication_verdict', 'N/A')}"
    )


# ============================================================
# TAB 4: WITHIN-CATEGORY DEEP DIVE
# ============================================================

def render_within_category(active_df, precomputed):
    """Render the Within-Category Deep Dive tab."""

    # Category selector
    all_categories = [
        'PressAccuracy', 'PressReasoning', 'Restating', 'Revoicing',
        'Marking', 'Context', 'KeepingTogether', 'GettingStudentsRelate',
    ]

    category = st.selectbox("Talk Move Category", all_categories)

    # If we have precomputed data for PressAccuracy and that's selected, show it
    tm = precomputed.get('talkmoves_results', {})
    s2 = tm.get('study2_within_category', {})

    if category == 'PressAccuracy' and s2:
        st.markdown(f"**N = {s2.get('total_utterances', 0):,} utterances** (full TalkMoves dataset)")
        _render_within_category_precomputed(s2, precomputed)
    elif active_df is not None:
        # Run analysis on the active transcript
        result = analyze_within_category(active_df, category)
        if result and result.get('total', 0) > 0:
            st.markdown(f"**N = {result['total']:,} utterances** (current transcript)")
            _render_within_category_live(result)
        else:
            st.info(f"No '{category}' utterances found in the current transcript.")
    else:
        st.info("Select a transcript or use pre-computed results for 'PressAccuracy'.")


def _render_within_category_precomputed(s2, precomputed):
    """Render within-category analysis from precomputed Study 2 data."""
    dist_col, grad_col = st.columns(2)
    wc_labels = ['A', 'B', 'C']
    wc_legend = ['A — Surface', 'B — Intermediate', 'C — Deep']
    wc_colors = ['#95a5a6', '#3498db', '#e74c3c']

    with dist_col:
        st.markdown("##### Distribution")
        wc_dist = s2.get('distribution', {})
        pcts = [
            wc_dist.get('level_A', {}).get('pct', 0),
            wc_dist.get('level_B', {}).get('pct', 0),
            wc_dist.get('level_C', {}).get('pct', 0),
        ]
        counts = [
            wc_dist.get('level_A', {}).get('count', 0),
            wc_dist.get('level_B', {}).get('count', 0),
            wc_dist.get('level_C', {}).get('count', 0),
        ]
        fig = _make_level_bar_chart(
            values=pcts,
            labels=wc_labels,
            legend_names=wc_legend,
            colors=wc_colors,
            y_title="% of Utterances",
            text_values=[f"{p}% (n={c:,})" for p, c in zip(pcts, counts)],
            y_range=(0, 115),
        )
        st.plotly_chart(fig, use_container_width=True)

    with grad_col:
        st.markdown("##### Student Evidence Gradient")
        val = s2.get('validation', {}).get('by_depth', {})
        a_pct = val.get('A', {}).get('evidence_pct', 0)
        b_pct = val.get('B', {}).get('evidence_pct', 0)
        c_pct = val.get('C', {}).get('evidence_pct', 0)
        ratio = round(c_pct / a_pct, 1) if a_pct > 0 else 0
        chi2 = s2.get('validation', {}).get('chi2', 0)
        v = s2.get('validation', {}).get('cramers_v', 0)
        fig = _make_level_bar_chart(
            values=[a_pct, b_pct, c_pct],
            labels=wc_labels,
            legend_names=wc_legend,
            colors=wc_colors,
            y_title="% Student Evidence",
            text_values=[f"{a_pct}%", f"{b_pct}%", f"{c_pct}%"],
            annotation=f"χ²={chi2}, V={v} — {ratio}× difference (A→C)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Example utterances
    st.markdown("---")
    st.markdown("##### Example Utterances by Depth")

    examples = s2.get('examples', {})
    ex_cols = st.columns(3)

    for i, (level, label, color) in enumerate([
        ('A', 'Surface', '#95a5a6'),
        ('B', 'Intermediate', '#3498db'),
        ('C', 'Deep', '#e74c3c'),
    ]):
        with ex_cols[i]:
            st.markdown(
                f'<h6 style="color: {color};">Level {level} ({label})</h6>',
                unsafe_allow_html=True,
            )
            for ex in examples.get(level, []):
                st.markdown(f'- "{ex}"')

    # LLM agreement stats
    llm = precomputed.get('llm_agreement', {})
    if llm:
        st.markdown("---")
        st.markdown("##### Convergent Validity (LLM)")
        agr = llm.get('agreement', {})
        llm_cols = st.columns(3)
        llm_cols[0].metric("Cohen's κ", f"{agr.get('cohens_kappa', 0):.3f}")
        llm_cols[1].metric("Agreement", f"{agr.get('agreement', 0)*100:.0f}%")
        llm_cols[2].metric("Model", llm.get('model', 'N/A'))

        rb_dist = agr.get('distribution', {}).get('rule_based', {})
        llm_dist = agr.get('distribution', {}).get('llm', {})
        rb_nonsurface = (rb_dist.get('B', 0) + rb_dist.get('C', 0)) / llm.get('n_coded', 200) * 100
        llm_nonsurface = (llm_dist.get('B', 0) + llm_dist.get('C', 0)) / llm.get('n_coded', 200) * 100
        st.caption(
            f"LLM finds {llm_nonsurface:.0f}% non-surface "
            f"(vs {rb_nonsurface:.0f}% rule-based) — "
            f"LLM sees *more* hidden depth, confirming that "
            f"rule-based estimates are conservative."
        )


def _render_within_category_live(result):
    """Render within-category analysis from live computation."""
    dist_col, grad_col = st.columns(2)
    wc_labels = ['A', 'B', 'C']
    wc_legend = ['A — Surface', 'B — Intermediate', 'C — Deep']
    wc_colors = ['#95a5a6', '#3498db', '#e74c3c']

    with dist_col:
        st.markdown("##### Distribution")
        dist = result['distribution']
        pcts = [dist['A']['pct'], dist['B']['pct'], dist['C']['pct']]
        fig = _make_level_bar_chart(
            values=pcts,
            labels=wc_labels,
            legend_names=wc_legend,
            colors=wc_colors,
            y_title="% of Utterances",
            text_values=[f"{p}%" for p in pcts],
            y_range=(0, 115),
        )
        st.plotly_chart(fig, use_container_width=True)

    with grad_col:
        st.markdown("##### Student Evidence Gradient")
        val = result.get('validation', {}).get('by_depth', {})
        if val:
            vals = [
                val.get('A', {}).get('evidence_pct', 0),
                val.get('B', {}).get('evidence_pct', 0),
                val.get('C', {}).get('evidence_pct', 0),
            ]
            chi2 = result.get('validation', {}).get('chi2', '')
            v = result.get('validation', {}).get('cramers_v', '')
            annotation = f"χ²={chi2}, V={v}" if chi2 else None
            fig = _make_level_bar_chart(
                values=vals,
                labels=wc_labels,
                legend_names=wc_legend,
                colors=wc_colors,
                y_title="% Student Evidence",
                text_values=[f"{v_}%" for v_ in vals],
                annotation=annotation,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Insufficient data for validation.")

    # Examples
    st.markdown("---")
    st.markdown("##### Example Utterances by Depth")
    examples = result.get('examples', {})
    ex_cols = st.columns(3)
    for i, (level, label, color) in enumerate([
        ('A', 'Surface', '#95a5a6'),
        ('B', 'Intermediate', '#3498db'),
        ('C', 'Deep', '#e74c3c'),
    ]):
        with ex_cols[i]:
            st.markdown(
                f'<h6 style="color: {color};">Level {level} ({label})</h6>',
                unsafe_allow_html=True,
            )
            for ex in examples.get(level, []):
                st.markdown(f'- "{ex}"')


# ============================================================
# TAB 5: ABOUT
# ============================================================

def render_about():
    """Render the About tab."""
    st.subheader("About DToM Lens")

    st.markdown("""
### The Double Theory of Mind (DToM) Framework

DToM identifies three layers of theory-of-mind relevant to educational AI systems:

| Layer | Description | Example |
|---|---|---|
| **L1** — Behavioral | Recognizing observable behavior | "Student said '9'" — classify as *Making a Claim* |
| **L2** — Intermediate | Modeling underlying processes | "Student is applying addition" — track *strategy use* |
| **L3** — Deep | Understanding *why* — reasoning and mental models | "Student thinks you can add numerators directly because they look like whole numbers" — probe *conceptual understanding* |

### Why It Matters

Current AI discourse classification tools (e.g., TalkMoves, M-Powering Teachers) primarily operate at
L1/L2, assigning behavioral labels to teacher and student talk. This is valuable for describing *what*
teachers do, but it **collapses meaningful variation** in *how deeply* teachers engage with student thinking.

Our analysis shows that within a single standard category like "Press for Accuracy":
- **87.8%** of utterances are surface-level (checking factual answers)
- **9.6%** are intermediate (requesting procedural explanation)
- **2.6%** are deep (probing conceptual understanding)

These levels predict student reasoning quality with a **4x difference** in evidence-providing rates
between surface and deep teacher moves.

### Methodology

**Datasets:**
- **TalkMoves** (Suresh et al., 2022): 567 K-12 math transcripts, 237,537 utterances.
  [GitHub](https://github.com/SumnerLab/TalkMoves) | CC BY-NC-SA 4.0
- **NCTE** (Demszky et al., 2021): 1,660 elementary math transcripts, 580,409 utterances.
  [GitHub](https://github.com/ddemszky/classroom-transcript-analysis) | Restricted access
  (individual application required; only aggregated statistics are shown in this tool)

**Classifier:** Rule-based linguistic pattern matching using established questioning taxonomies
(Chin 2007, Franke et al. 2009, Boaler & Brodie 2004). Validated against independent LLM coding
(Cohen's kappa = 0.229; LLM finds *more* depth, confirming rule-based estimates are conservative).

**Statistical approach:** Chi-square tests for sequential patterns, Pearson/Spearman correlations for
transcript-level relationships, Cohen's d for effect sizes, cross-dataset replication.

### Citation

If you use DToM Lens in your research, please cite:

> [Anonymous]. (2026). DToM Lens: An Interactive Diagnostic Tool for Mentalizing Depth
> in Teacher Discourse. *Proceedings of the ACM Conference on Learning @ Scale (L@S '26)*.

### Links

- Paper: *Under review*
- Source code: *Available upon publication*
    """)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Load data
    precomputed = load_precomputed()
    sample_transcripts = load_sample_transcripts()

    # Sidebar
    data_source, active_transcript, active_df = render_sidebar(
        precomputed, sample_transcripts
    )

    # Main area tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Transcript Explorer",
        "Comparison View",
        "Dashboard",
        "Within-Category",
        "About",
    ])

    with tab1:
        if data_source == "Pre-computed results":
            st.info("Pre-computed results selected — switch to the **Dashboard** tab to view aggregate statistics.")
        else:
            render_transcript_explorer(active_transcript, active_df)

    with tab2:
        if data_source == "Pre-computed results":
            st.info("Pre-computed results selected — switch to the **Dashboard** tab to view aggregate statistics.")
        else:
            render_comparison_view(active_transcript, active_df)

    with tab3:
        render_dashboard(precomputed)

    with tab4:
        render_within_category(active_df, precomputed)

    with tab5:
        render_about()


if __name__ == '__main__':
    main()
