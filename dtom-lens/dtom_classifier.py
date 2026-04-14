"""
DToM Classifier Module
======================
Rule-based mentalizing depth classifier for teacher discourse.

Ported from the DToM empirical analysis pipeline.
Classifies teacher utterances at two levels:
  1. L3 Depth: Maps standard talk move categories to mentalizing depth (0/1/2)
  2. Within-category depth: Linguistic markers → A (surface) / B (intermediate) / C (deep)
"""

import re

import pandas as pd

# ============================================================
# L3 DEPTH MAPPING
# ============================================================

L3_DEPTH_MAP = {
    # Level 0 — No L3 (procedural/management)
    'None':               0,
    'Context':            0,
    'KeepingTogether':    0,
    'Marking':            0,
    # Level 1 — Surface L3 (acknowledge/check)
    'Restating':          1,
    'Revoicing':          1,
    'PressAccuracy':      1,
    # Level 2 — Deep L3 (probe reasoning)
    'PressReasoning':     2,
    'GettingStudentsRelate': 2,
}

L3_LABELS = {
    0: 'L1 Behavioral (procedural/management)',
    1: 'L2 Intermediate (acknowledge/check)',
    2: 'L3 Deep (probe reasoning)',
}

# ============================================================
# WITHIN-CATEGORY MENTALIZING DEPTH: LINGUISTIC MARKERS
# ============================================================

DEEP_PATTERNS = [
    r'\bwhy\b.*\b(think|work|true|right|correct|happen|say|choose|pick)\b',
    r'\bwhy\b(?!.*\bnot\b)',
    r'\bhow do you know\b',
    r'\bhow can you tell\b',
    r'\bwhat makes you\b',
    r'\bwhat tells you\b',
    r'\bexplain\b.*\b(thinking|reasoning|why|how)\b',
    r'\bwhat would happen\b',
    r'\bwhat if\b',
    r'\bdoes that make sense\b',
    r'\bdoes that work\b',
    r'\bprove\b',
    r'\bjustify\b',
    r'\bconvince\b',
    r'\bhow does that connect\b',
    r'\bhow does that relate\b',
    r'\bdo you agree\b.*\bwhy\b',
    r'\bwhy do you agree\b',
    r'\bis there another way\b',
    r'\bcould you do it differently\b',
    r'\bwhat.{0,20}mean\b.*\bby\b',
    r'\btell me more about\b.*\bthinking\b',
]

INTERMEDIATE_PATTERNS = [
    r'\bhow did you\b',
    r'\bhow do you\b',
    r'\bwhat did you do\b',
    r'\bwhat.{0,10}you do\b',
    r'\bshow\s+(me|us)\b',
    r'\bexplain\b',
    r'\btell\s+(me|us)\s+how\b',
    r'\btell\s+(me|us)\s+what\b',
    r'\bwhat.{0,10}(method|strategy|approach|steps|process)\b',
    r'\bwalk.{0,10}through\b',
    r'\bdescribe\b',
    r'\bwhat.{0,10}(next|then)\b',
    r'\bcan you\s+(show|tell|explain|describe)\b',
    r'\bhow.{0,15}(figure|solve|find|get|work)\b',
]

MENTAL_DEPTH_LABELS = {
    'A': 'Surface (checking answer/fact)',
    'B': 'Intermediate (requesting procedure)',
    'C': 'Deep (probing understanding)',
}

DEPTH_COLORS = {
    'A': '#95a5a6',  # gray
    'B': '#3498db',  # blue
    'C': '#e74c3c',  # red
}

L3_COLORS = {
    0: '#95a5a6',  # gray
    1: '#3498db',  # blue
    2: '#e74c3c',  # red
}


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_teacher_tag(tag: str) -> str | None:
    """Normalize TalkMoves tag variants to canonical categories."""
    if pd.isna(tag):
        return None
    tag = str(tag).strip().lower()
    if 'none' in tag or tag in ['1', '2', '']:
        return 'None'
    if 'keeping' in tag:
        return 'KeepingTogether'
    if 'getting' in tag or 'relate' in tag:
        return 'GettingStudentsRelate'
    if 'restating' in tag:
        return 'Restating'
    if 'revoicing' in tag:
        return 'Revoicing'
    if 'marking' in tag:
        return 'Marking'
    if 'context' in tag:
        return 'Context'
    if 'reasoning' in tag:
        return 'PressReasoning'
    if 'accuracy' in tag:
        return 'PressAccuracy'
    return 'Other'


def normalize_student_tag(tag: str) -> str | None:
    """Normalize student tag variants."""
    if pd.isna(tag):
        return None
    tag = str(tag).strip().lower()
    if 'none' in tag or tag in ['1', '2', '5', '\\']:
        return 'None'
    if 'relating' in tag or 'relate' in tag:
        return 'RelatingToAnother'
    if 'asking' in tag or 'information' in tag:
        return 'AskingForInfo'
    if 'claim' in tag:
        return 'MakingClaim'
    if 'evidence' in tag or 'reasoning' in tag:
        return 'ProvidingEvidence'
    return 'Other'


# ============================================================
# CLASSIFIER FUNCTIONS
# ============================================================

def classify_mentalizing_depth(text: str) -> str:
    """Classify a teacher utterance's mentalizing depth using linguistic markers.

    Returns:
        'A' (Surface): Checking factual answers, requesting numbers/results
        'B' (Intermediate): Asking for procedure/method explanation
        'C' (Deep): Probing conceptual understanding, asking why/justify

    Priority: Deep > Intermediate > Surface (default).
    """
    text = str(text).lower().strip()

    for pattern in DEEP_PATTERNS:
        if re.search(pattern, text):
            return 'C'

    for pattern in INTERMEDIATE_PATTERNS:
        if re.search(pattern, text):
            return 'B'

    return 'A'


def get_l3_depth(talk_move: str) -> int | None:
    """Map talk move category to L3 depth level (0, 1, 2)."""
    return L3_DEPTH_MAP.get(talk_move)


# ============================================================
# DATA PROCESSING
# ============================================================

def process_transcript(df: pd.DataFrame) -> pd.DataFrame:
    """Process a raw transcript DataFrame: normalize tags, assign depths.

    Expects columns: 'Sentence', 'Speaker', and optionally
    'Teacher Tag', 'Student Tag'.

    Also accepts lowercase variants: 'text'/'sentence', 'speaker',
    'teacher_tag'/'tag', 'student_tag'.
    """
    df = df.copy()

    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ('sentence', 'text'):
            col_map[col] = 'Sentence'
        elif cl == 'speaker':
            col_map[col] = 'Speaker'
        elif cl in ('teacher tag', 'teacher_tag'):
            col_map[col] = 'Teacher Tag'
        elif cl in ('student tag', 'student_tag'):
            col_map[col] = 'Student Tag'
    df = df.rename(columns=col_map)

    # Normalize tags if present
    if 'Teacher Tag' in df.columns:
        df['t_move'] = df['Teacher Tag'].apply(normalize_teacher_tag)
        df['l3_depth'] = df['t_move'].map(L3_DEPTH_MAP)
    if 'Student Tag' in df.columns:
        df['s_move'] = df['Student Tag'].apply(normalize_student_tag)

    # Classify within-category depth for teacher utterances
    if 't_move' in df.columns:
        teacher_mask = df['t_move'].notna()
        df.loc[teacher_mask, 'mental_depth'] = df.loc[teacher_mask, 'Sentence'].apply(
            classify_mentalizing_depth
        )

    # Word count
    df['word_count'] = df['Sentence'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )

    return df
