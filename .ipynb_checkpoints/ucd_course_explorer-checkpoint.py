"""
===============================================================================
PROGRAMME & MODULE EXPLORER — STREAMLIT APPLICATION (Single-file)
===============================================================================

Purpose
-------
This Streamlit application provides an interactive explorer for university
programmes (majors) and their associated modules. It is designed for fast,
keyboard-driven browsing and inspection of curriculum structure and module
relationships (eligibility constraints and similarity).

Layout (Fixed)
--------------
• Sidebar (Column 1): Programme/Major search with paginated results.
• Main Column 2:      Module list.
    - If a major is selected: shows staged modules for that major + in-major search.
    - If no major selected:  global module search across all modules.
• Main Column 3:      Module details pane:
    - Header + subtitle
    - Description expander
    - Eligibility expander (prereq/coreq/incompatible/learning requirements)
    - Similar modules expander
    - Other majors expander

Data Inputs (Fixed)
-------------------
The app reads only the pre-built parquet artifacts produced by the 1900_ pipeline:
  • data/streamlit_major_results.parquet
  • data/streamlit_major_meta.parquet
  • data/streamlit_modules_by_major.parquet
  • data/streamlit_module_details.parquet

UI Freeze
---------
The UI is intentionally frozen:
  • same widgets, same layout, same CSS, same session_state semantics
  • no new controls, no removed controls

Internal changes included (1–5)
-------------------------------
1) Cache-busting signature for parquet artifacts:
   - The cached loader is keyed by file modification times so updated parquet files
     are picked up without restarting the Streamlit server.

2) Import hygiene without behavioural change:
   - Keep imports that are present in the app environment even if not used.

3) Prevent button key collisions in module lists:
   - Add a stable index suffix to keys in the eligibility/similarity list renderer.

4) Normalize module search queries consistently:
   - In-major module search uses (strip + lower) to match global search semantics.

5) Subtitle fallback avoids stale major context:
   - When Column 3 must synthesize a subtitle (because one wasn’t explicitly provided
     during selection), it includes the current selected major only if that major
     actually contains the module (using module_to_majors).

===============================================================================
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # intentionally kept (no UI effect)

# =============================================================================
# RESET SUPPORT
# =============================================================================
# The reset mechanism is a two-step process:
#   (a) A button sets a session_state flag and triggers a rerun.
#   (b) At the top of the script (before any widgets), the flag is detected and
#       session_state is cleared.
#
# The epoch counter is used in widget keys so that text inputs reset reliably
# when session_state is cleared (Streamlit otherwise tries to preserve widget values).
_RESET_FLAG = "__DO_RESET_APP__"
_WIDGET_EPOCH = "__WIDGET_EPOCH__"


def _reset_app_state(*, hard: bool = False) -> None:
    """
    Reset session_state to a clean slate.

    Parameters
    ----------
    hard:
        If True, clears Streamlit caches (data/resource). This is optional and
        not used by the normal reset button because cache clears slow down reruns.
        If False, only session_state is reset.
    """
    if hard:
        st.cache_data.clear()
        st.cache_resource.clear()

    # Increment the epoch so widget keys change on next render.
    next_epoch = int(st.session_state.get(_WIDGET_EPOCH, 0)) + 1

    # Clear all session_state keys and seed the new epoch.
    st.session_state.clear()
    st.session_state[_WIDGET_EPOCH] = next_epoch


# This must run before any widgets are created.
if st.session_state.get(_RESET_FLAG, False):
    _reset_app_state(hard=False)
    st.rerun()

# =============================================================================
# STREAMLIT CONFIG (Fixed)
# =============================================================================
st.set_page_config(
    layout="wide",
    page_title="Programme Explorer",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS (Fixed)
# =============================================================================
# CSS controls:
#   • fixed sidebar width
#   • fixed width for main Column 2
#   • fluid width for main Column 3
#   • two-line button styling for lists
#   • spacing tweaks (page top padding, expander spacing, etc.)
def _inject_css() -> None:
    """Inject fixed CSS used to control sizing/spacing and the two-line button style."""
    st.markdown(
        """
        <style>
        /* 1. SIDEBAR WIDTH (Fixed 500px) */
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 500px !important;
        }

        /* 2. MAIN COLUMN LAYOUT (prevent wrapping) */
        .block-container div[data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
        }

        /* Column 2: fixed width */
        .block-container div[data-testid="stHorizontalBlock"] > div:first-child {
            width: 500px !important;
            min-width: 500px !important;
            max-width: 500px !important;
            flex: 0 0 500px !important;
        }

        /* Column 3: fluid remainder */
        .block-container div[data-testid="stHorizontalBlock"] > div:last-child {
            flex: 1 1 auto !important;
            min-width: 0px !important;
            width: auto !important;
        }

        /* Ensure internal column content fills its container */
        .block-container div[data-testid="stHorizontalBlock"] > div [data-testid="column"] {
            width: 100% !important;
        }

        /* 3. Top padding reduction */
        .block-container {
            padding-top: 2.7rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* 4. Sidebar top spacing removal */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:first-child {
            padding-top: 0rem !important;
            margin-top: -1rem !important;
        }

        /* 5. Header styling */
        h2 {
            font-size: 1rem !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            margin-top: 0px !important;
            margin-bottom: 10px !important;
            padding-top: 0px !important;
            color: rgb(49, 51, 63) !important;
            min-height: 0px !important;
        }

        /* Column 3 subtitle spacing */
        .block-container div[data-testid="stHorizontalBlock"]
          > div:last-child
          .col3-subtitle {
            margin-top: -36px !important;
            margin-bottom: -12px !important;
            font-size: 0.875rem !important;
            color: rgba(49, 51, 63, 0.7) !important;
          }

        /* Reduce space above expander in Column 3 */
        .block-container div[data-testid="stHorizontalBlock"]
          > div:last-child
          div[data-testid="stExpander"] {
            margin-top: -16px !important;
        }

        /* 6. Two-line list button style (primary/secondary) */
        button[kind="secondary"], button[kind="primary"] {
            width: 100% !important;
            text-align: left !important;
            justify-content: flex-start !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 4px !important;
            margin-bottom: -15px !important;

            height: auto !important;
            min-height: 50px !important;
            white-space: pre-wrap !important;
            line-height: 1.3 !important;
            padding: 8px 10px !important;
        }

        /* Secondary buttons: no background */
        button[kind="secondary"] {
            background: transparent !important;
        }

        /* Subtitle/base font */
        button[kind="secondary"] p, button[kind="primary"] p {
            font-size: 12px !important;
            font-weight: 400 !important;
            color: #666 !important;
        }

        /* Title (first line) styling */
        button[kind="secondary"] p::first-line {
            font-size: 14px !important;
            font-weight: 600 !important;
            color: #31333F !important;
        }

        /* Active state (primary) */
        button[kind="primary"] {
            background: #e6f3ff !important;
        }
        button[kind="primary"] p {
            color: #4d94ff !important;
        }
        button[kind="primary"] p::first-line {
            color: #0066cc !important;
        }

        button[kind="secondary"]:hover {
            background: #f0f2f6 !important;
        }

        /* 7. Sidebar pagination button reset */
        [data-testid="stSidebar"] div[data-testid="column"] button {
            border: 1px solid #ddd !important;
            background: transparent !important;
            text-align: center !important;
            justify-content: center !important;
            margin-bottom: 0px !important;
            padding: 8px 15px !important;
            min-height: 0px !important;
            height: auto !important;
        }

        [data-testid="stSidebar"] div[data-testid="column"] button p {
            font-size: 14px !important;
            color: #31333F !important;
            font-weight: 400 !important;
        }

        [data-testid="stSidebar"] div[data-testid="column"] button:hover {
            border-color: #ff4b4b !important;
            color: #ff4b4b !important;
            background: transparent !important;
        }

        [data-testid="stSidebar"] div[data-testid="column"]:first-child .stButton {
            display: flex;
            justify-content: flex-start;
        }
        [data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton {
            display: flex;
            justify-content: flex-end;
        }

        /* 8. Input padding */
        .stTextInput input {
            padding: 8px 10px;
        }

        /* Prevent clipping from negative button margins */
        .block-container div[data-testid="stHorizontalBlock"] > div:first-child {
            padding-bottom: 20px !important;
        }

        /* Reduce vertical space above stage radios */
        div[data-testid="stRadio"] {
            margin-top: -2rem;
        }

        /* Expander heading spacing in Column 3 */
        .block-container div[data-testid="stHorizontalBlock"]
          > div:last-child
          div[data-testid="stExpander"]
          h2 {
            margin-top: 12px !important;
            margin-bottom: -36px !important;
        }

        .block-container div[data-testid="stHorizontalBlock"]
          > div:last-child
          div[data-testid="stExpander"] > div {
            padding-top: 0px !important;
        }

        /* Sidebar forms: remove frame/indent */
        [data-testid="stSidebar"] form {
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            background: transparent !important;
        }

        [data-testid="stSidebar"] div[data-testid="stForm"] {
            padding: 0 !important;
            margin: 0 !important;
        }

        [data-testid="stSidebar"] .stTextInput {
            margin-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_css()

# =============================================================================
# DATA LOADING + INDEX BUILDING
# =============================================================================
# The app is designed to be rerun frequently (every widget interaction triggers a rerun).
# To keep reruns fast, parquet reads and derived lookups are cached.
#
# Change (1): cache-busting signature
# -----------------------------------
# Streamlit's cache keys depend on function inputs. We compute a small signature
# from the parquet modification times and pass that into the cached loader. When
# files are regenerated, mtimes change, so the cache is invalidated automatically.
DATA_DIR = Path("data")
ARTIFACT_PATHS = {
    "major_results": DATA_DIR / "streamlit_major_results.parquet",
    "major_meta": DATA_DIR / "streamlit_major_meta.parquet",
    "mods_by_major": DATA_DIR / "streamlit_modules_by_major.parquet",
    "module_details": DATA_DIR / "streamlit_module_details.parquet",
}


def _artifact_signature() -> tuple[int, int, int, int]:
    """
    Return a deterministic signature of the current artifact set.

    The signature is a tuple of nanosecond modification times.
    If any artifact file is rewritten, its mtime_ns changes, changing this tuple.
    """
    return (
        ARTIFACT_PATHS["major_results"].stat().st_mtime_ns,
        ARTIFACT_PATHS["major_meta"].stat().st_mtime_ns,
        ARTIFACT_PATHS["mods_by_major"].stat().st_mtime_ns,
        ARTIFACT_PATHS["module_details"].stat().st_mtime_ns,
    )


def _require_cols(name: str, df: pd.DataFrame, required: set[str]) -> None:
    """Fail fast with a clear error if a required column is missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def load_artifacts_and_indexes(signature: tuple[int, int, int, int]):
    """
    Load parquet artifacts and build all lookup structures needed for O(1) UI actions.

    Parameters
    ----------
    signature:
        Cache-busting key (artifact mtimes). It is not used directly in the logic;
        it exists solely to invalidate the cache when parquet files change.

    Returns
    -------
    A tuple containing:
      • normalized DataFrames used for filtering and display
      • lookup dictionaries and indices for fast selection
      • global counts for UI labels
    """
    # --- Load the parquet files ---
    df_major_results = pd.read_parquet(ARTIFACT_PATHS["major_results"])
    df_major_meta = pd.read_parquet(ARTIFACT_PATHS["major_meta"])
    df_mods_by_major = pd.read_parquet(ARTIFACT_PATHS["mods_by_major"])
    df_module_details = pd.read_parquet(ARTIFACT_PATHS["module_details"])

    # --- Validate schemas (fixed set required by UI logic) ---
    _require_cols(
        "streamlit_major_results.parquet",
        df_major_results,
        {"major_code", "result_title", "result_subtitle", "search_blob"},
    )
    _require_cols(
        "streamlit_major_meta.parquet",
        df_major_meta,
        {
            "major_code",
            "programme_title",
            "programme_level",
            "programme_award",
            "programme_duration",
            "programme_attendance",
            "programme_url",
        },
    )
    _require_cols(
        "streamlit_modules_by_major.parquet",
        df_mods_by_major,
        {
            "major_code",
            "module_code",
            "module_stage",
            "module_type",
            "module_level",
            "result_title",
            "result_subtitle",
            "search_blob",
            "sort_stage",
            "sort_type",
            "sort_level",
        },
    )
    _require_cols(
        "streamlit_module_details.parquet",
        df_module_details,
        {
            "module_code",
            "module_title",
            "module_description",
            "module_trimester",
            "module_credits",
            "module_level",
            "module_stage",
            "module_type",
            "module_coordinator_name",
            "has_prerequisite_modules",
            "has_corequisite_modules",
            "has_incompatible_modules",
            "has_learning_requirement_modules",
            "prerequisite_module_for",
            "corequisite_module_for",
            "learning_requirement_module_for",
            "top_n_modules_same_school",
            "top_n_modules_different_school",
        },
    )

    # --- Normalize types used in filtering and display ---
    # Normalize key columns to strings (consistent lookup keys everywhere).
    df_major_results = df_major_results.copy()
    df_major_results["major_code"] = df_major_results["major_code"].astype(str)

    # Programme search is substring matching against search_blob; normalize to lowercase once.
    df_major_results["search_blob"] = df_major_results["search_blob"].fillna("").astype(str).str.lower()

    # Ensure button labels are always present as strings.
    df_major_results["result_title"] = df_major_results["result_title"].fillna("").astype(str)
    df_major_results["result_subtitle"] = df_major_results["result_subtitle"].fillna("").astype(str)

    df_major_meta = df_major_meta.copy()
    df_major_meta["major_code"] = df_major_meta["major_code"].astype(str)
    for c in [
        "programme_title",
        "programme_level",
        "programme_award",
        "programme_duration",
        "programme_attendance",
        "programme_url",
    ]:
        df_major_meta[c] = df_major_meta[c].fillna("").astype(str)

    df_mods_by_major = df_mods_by_major.copy()
    df_mods_by_major["major_code"] = df_mods_by_major["major_code"].astype(str)
    df_mods_by_major["module_code"] = df_mods_by_major["module_code"].astype(str)
    df_mods_by_major["search_blob"] = df_mods_by_major["search_blob"].fillna("").astype(str).str.lower()
    df_mods_by_major["result_title"] = df_mods_by_major["result_title"].fillna("").astype(str)
    df_mods_by_major["result_subtitle"] = df_mods_by_major["result_subtitle"].fillna("").astype(str)

    # Stage selector uses a numeric stage value; normalize once for fast filtering.
    df_mods_by_major["_stage_int"] = (
        pd.to_numeric(df_mods_by_major["sort_stage"], errors="coerce").fillna(99).astype(int)
    )

    df_module_details = df_module_details.copy()
    df_module_details["module_code"] = df_module_details["module_code"].astype(str)
    for c in ["module_title", "module_trimester", "module_coordinator_name", "module_description"]:
        df_module_details[c] = df_module_details[c].fillna("").astype(str)

    # -------------------------------------------------------------------------
    # LOOKUPS / INDEXES
    # -------------------------------------------------------------------------
    # The UI relies on constant-time lookups for most interactions:
    #
    #   • major_title_lookup: major_code -> programme_title
    #   • major_search_index: Series indexed by major_code -> search_blob
    #   • modules_by_major:   major_code -> DataFrame of modules (long table subset)
    #   • module_details_lookup: module_code -> dict of module details
    #   • module_label_lookup_global: module_code -> (result_title, result_subtitle)
    #   • major_meta_lookup:  major_code -> meta dict (for “Other majors” labels)
    #   • module_to_majors:   module_code -> list of major_codes (for “Other majors” and subtitle safety)
    major_title_lookup = dict(zip(df_major_meta["major_code"], df_major_meta["programme_title"]))

    major_search_index = (
        df_major_results.drop_duplicates(subset=["major_code"]).set_index("major_code")["search_blob"]
    )

    modules_by_major = {
        major_code: g.reset_index(drop=True)
        for major_code, g in df_mods_by_major.groupby("major_code", sort=False)
    }

    module_details_lookup = df_module_details.set_index("module_code", drop=False).to_dict(orient="index")

    m = df_mods_by_major.drop_duplicates(subset=["module_code"])
    module_label_lookup_global = dict(zip(m["module_code"], zip(m["result_title"], m["result_subtitle"])))

    meta_cols = [
        "major_code",
        "programme_title",
        "programme_level",
        "programme_award",
        "programme_duration",
        "programme_attendance",
    ]
    major_meta_lookup = df_major_meta[meta_cols].set_index("major_code", drop=False).to_dict(orient="index")

    tmp = df_mods_by_major[["module_code", "major_code"]].drop_duplicates()
    module_to_majors: dict[str, list[str]] = {}
    for module_code, g in tmp.groupby("module_code", sort=False):
        module_to_majors[str(module_code)] = sorted(set(g["major_code"].astype(str).tolist()))

    # Counts are used only for captions/placeholders.
    n_majors = int(df_major_results["major_code"].nunique())
    n_modules = int(df_module_details["module_code"].nunique())

    return (
        df_major_results,
        df_major_meta,
        df_mods_by_major,
        df_module_details,
        major_title_lookup,
        major_search_index,
        modules_by_major,
        module_details_lookup,
        module_label_lookup_global,
        major_meta_lookup,
        module_to_majors,
        n_majors,
        n_modules,
    )


# -----------------------------------------------------------------------------
# Load all data and lookup structures (cached; invalidated by signature changes).
# -----------------------------------------------------------------------------
(
    df_major_results,
    df_major_meta,
    df_mods_by_major,
    df_module_details,
    major_title_lookup,
    major_search_index,
    modules_by_major,
    module_details_lookup,
    module_label_lookup_global,
    major_meta_lookup,
    module_to_majors,
    N_MAJORS,
    N_MODULES,
) = load_artifacts_and_indexes(_artifact_signature())

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# The UI logic reads and writes the following keys. Defaults are defined here so
# the remainder of the code can assume keys exist.
_defaults = {
    "selected_major_code": None,
    "selected_major_title": None,
    "selected_major_subtitle": None,
    "selected_module_code": None,
    "selected_module_title": None,
    "selected_module_subtitle": None,
    "page_number": 0,
    "last_query": "",
    "forced_major_filter": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# SELECTION HANDLERS
# =============================================================================
# These functions centralize session_state mutations related to selections.
# Buttons in the sidebar and module lists call these, then trigger st.rerun().
def handle_programme_selection(code, title, subtitle=None):
    """
    Select a programme/major for browsing in Column 2.

    Selecting a programme resets the currently selected module so Column 3 clears.
    """
    if st.session_state.selected_major_code != code:
        st.session_state.selected_major_code = code
        st.session_state.selected_major_title = title
        st.session_state.selected_major_subtitle = subtitle if subtitle is not None else None

        st.session_state.selected_module_code = None
        st.session_state.selected_module_title = None
        st.session_state.selected_module_subtitle = None


def handle_module_selection(code, *, major_code=None, title=None, subtitle=None):
    """
    Select a module for display in Column 3.

    Parameters
    ----------
    code:
        The module_code to display.
    major_code:
        Optional. If provided, it activates that major in Column 2 (programme mode).
        This is used when selecting a module from within a programme list.
        In global module search mode, major_code is intentionally not set.
    title, subtitle:
        Optional display strings used for the Column 3 subtitle. If not provided,
        Column 3 can synthesize a subtitle from module details.
    """
    code = str(code)

    if st.session_state.selected_module_code != code:
        st.session_state.selected_module_code = code

    if title is not None:
        st.session_state.selected_module_title = str(title)

    if subtitle is not None:
        st.session_state.selected_module_subtitle = str(subtitle)

    # If a major_code is provided, the module selection also activates that major.
    if major_code is not None:
        maj = str(major_code)
        if maj.strip():
            st.session_state.selected_major_code = maj
            st.session_state.selected_major_title = major_title_lookup.get(
                maj, st.session_state.selected_major_title
            )

# =============================================================================
# SMALL TEXT HELPERS
# =============================================================================
def _plural(n: int, singular: str, plural: str | None = None) -> str:
    """Format simple singular/plural phrases used in captions."""
    if n == 1:
        return f"1 {singular}"
    return f"{n:,} {plural or singular + 's'}"

# =============================================================================
# RENDERERS
# =============================================================================
def render_module_list_section(*, module_codes, header_text, button_key_prefix, n: int | None = None):
    """
    Render a vertical list of module buttons under a heading.

    This renderer is used in Column 3 for:
      • eligibility lists (prereqs, coreqs, etc.)
      • similarity lists

    module_codes may be:
      • list[str]
      • numpy array
      • list[tuple[str, ...]]  (e.g., (code, score) pairs) — only the first element is used.

    Change (3): avoid key collisions by adding an index suffix to button keys.
    """
    # Normalize input containers.
    if isinstance(module_codes, np.ndarray):
        module_codes = module_codes.tolist()

    # If list items are tuples/lists, treat as (code, ...) and keep only the code.
    if isinstance(module_codes, list) and module_codes and isinstance(module_codes[0], (list, tuple)):
        module_codes = [m[0] for m in module_codes]

    if not isinstance(module_codes, list) or len(module_codes) == 0:
        return

    # Optional truncation for “Top-N” style lists.
    if n is not None:
        module_codes = module_codes[:n]

    st.header(header_text.format(n=len(module_codes)))

    # module_code -> (title, subtitle) labels come from the global long table.
    label_lookup = module_label_lookup_global

    for idx, code in enumerate(module_codes):
        code = str(code)

        if code in label_lookup:
            title_line, subtitle_line = label_lookup[code]
            subtitle_line = subtitle_line if str(subtitle_line).strip() else " "
        else:
            title_line, subtitle_line = code, " "

        label = f"{title_line}\n{subtitle_line}"

        # Change (3): include idx in key to guarantee uniqueness even if duplicates appear.
        if st.button(label, key=f"{button_key_prefix}_{code}_{idx}", type="secondary"):
            handle_module_selection(code, title=title_line, subtitle=subtitle_line)
            st.rerun()


def render_col2_stage_module_list(*, mods: pd.DataFrame, selected_major_code: str, activate_major_on_select: bool = True):
    """
    Render the staged module list in Column 2.

    The list is grouped by stage via a horizontal radio selector.
    Within each stage:
      • core modules appear first
      • a divider is shown between core and option modules

    Parameters
    ----------
    mods:
        DataFrame containing module rows to display.
    selected_major_code:
        Used to key the stage selector widget (each mode has its own selector state).
    activate_major_on_select:
        If True, selecting a module activates its major (programme mode).
        If False, selecting a module does NOT activate a major (global search mode).
    """
    if mods.empty:
        return False

    stage_col = mods["_stage_int"].to_numpy()
    stages = np.unique(stage_col)
    stages = stages[stages != 99]
    if stages.size == 0:
        return False
    stages = stages.tolist()

    st.write(" ")

    selected_stage = st.radio(
        label="Stage selector",
        label_visibility="collapsed",
        options=stages,
        format_func=lambda s: f"Stage {s}",
        horizontal=True,
        key=f"col2_stage_selector_{selected_major_code}",
    )

    stage_df = mods.loc[mods["_stage_int"] == int(selected_stage)]

    seen_core = False
    divider_shown = False

    for r in stage_df.itertuples(index=False):
        module_type = str(getattr(r, "module_type", "")).strip().lower()

        if module_type == "option" and seen_core and not divider_shown:
            st.markdown(
                "<hr style='margin: 0.75rem 0; border: none; border-top: 1px solid #e0e0e0;'/>",
                unsafe_allow_html=True,
            )
            divider_shown = True

        if module_type == "core":
            seen_core = True

        subtitle = getattr(r, "result_subtitle", "")
        subtitle = subtitle if str(subtitle).strip() else " "
        title_line = getattr(r, "result_title", "") or getattr(r, "module_code", "")
        label = f"{title_line}\n{subtitle}"

        mod_code = getattr(r, "module_code")
        maj_code = getattr(r, "major_code", None)

        if st.button(
            label,
            key=f"btn_mod_{mod_code}_{(maj_code or 'ALL')}_{selected_stage}",
            type="secondary",
        ):
            handle_module_selection(
                mod_code,
                major_code=(maj_code if activate_major_on_select else None),
                title=getattr(r, "result_title", None),
                subtitle=getattr(r, "result_subtitle", None),
            )
            st.rerun()

# =============================================================================
# SIDEBAR (Column 1): PROGRAMME SEARCH + PAGINATION + RESET
# =============================================================================
with st.sidebar:
    epoch = int(st.session_state.get(_WIDGET_EPOCH, 0))

    st.header("Major/Programme Search")

    prog_search = st.text_input(
        "Search Programmes",
        label_visibility="collapsed",
        placeholder=f"Search all {N_MAJORS:,} majors",
        key=f"prog_search_box_{epoch}",
    )

    # When the query changes, reset pagination and clear any forced programme filter.
    if prog_search != st.session_state.last_query:
        st.session_state.page_number = 0
        st.session_state.last_query = prog_search
        st.session_state.forced_major_filter = None

    # Determine the set of programmes to display:
    #   • If forced_major_filter is set, show only that programme row.
    #   • Otherwise filter by search substring over the precomputed search index.
    if st.session_state.forced_major_filter:
        all_matches = df_major_results.loc[
            df_major_results["major_code"] == st.session_state.forced_major_filter
        ]
    else:
        q = (prog_search or "").strip().lower()
        if q:
            matching_codes = major_search_index[
                major_search_index.str.contains(q, na=False, regex=False)
            ].index
            all_matches = df_major_results[df_major_results["major_code"].isin(matching_codes)]
        else:
            all_matches = df_major_results

    # Pagination and current page slice.
    items_per_page = 10
    total_matches = len(all_matches)
    total_pages = math.ceil(total_matches / items_per_page) if total_matches else 1
    current_page = st.session_state.page_number

    start_idx = current_page * items_per_page
    end_idx = start_idx + items_per_page
    current_batch = all_matches.iloc[start_idx:end_idx]
    display_end = min(end_idx, total_matches)

    # Result-count caption.
    if total_matches == 0:
        st.caption("No matches found")
    elif total_matches <= items_per_page:
        st.caption("Showing 1 major" if total_matches == 1 else f"Showing {total_matches} majors")
    else:
        st.caption(f"Showing {start_idx + 1}..{display_end} of {total_matches} majors")

    # Programme selection buttons (two-line labels).
    for r in current_batch.itertuples(index=False):
        subtitle = getattr(r, "result_subtitle", "")
        subtitle = subtitle if str(subtitle).strip() else " "
        label = f"{getattr(r, 'result_title')}\n{subtitle}"

        maj_code = str(getattr(r, "major_code"))
        if st.button(label, key=f"btn_prog_{maj_code}", type="secondary"):
            st.session_state.forced_major_filter = None
            programme_title = major_title_lookup.get(maj_code, "")
            programme_subtitle = getattr(r, "result_subtitle", "") or ""
            handle_programme_selection(maj_code, programme_title, programme_subtitle)
            st.rerun()

    # Prev/Next pagination arrows.
    if total_pages > 1:
        st.write("")
        col_prev, _, col_next = st.columns([3, 5, 3])

        with col_prev:
            if current_page > 0:
                if st.button("◀", key="btn_prev", use_container_width=False):
                    st.session_state.page_number -= 1
                    st.rerun()

        with col_next:
            if current_page < total_pages - 1:
                if st.button("▶", key="btn_next", use_container_width=False):
                    st.session_state.page_number += 1
                    st.rerun()

    # Reset button. The actual reset is performed at the top of the script.
    st.write("")
    if st.button("↺ Reset", key="btn_reset_app"):
        st.session_state[_RESET_FLAG] = True
        st.rerun()

# =============================================================================
# MAIN LAYOUT (Columns 2 and 3)
# =============================================================================
col_modules, col_details = st.columns([1, 1])

# =============================================================================
# COLUMN 2: MODULE LIST (programme mode OR global search mode)
# =============================================================================
epoch = int(st.session_state.get(_WIDGET_EPOCH, 0))

with col_modules:
    # Programme mode: a major has been selected via the sidebar or via "Other majors".
    if st.session_state.selected_major_code:
        st.header(f"{st.session_state.selected_major_code} - {st.session_state.selected_major_title}")

        mod_search = st.text_input(
            "Search Modules",
            label_visibility="collapsed",
            placeholder=f"Search modules in {st.session_state.selected_major_code}...",
            key=f"mod_search_box_major_{epoch}",
        )

        mods = modules_by_major.get(st.session_state.selected_major_code, pd.DataFrame())

        # Change (4): normalize in-major query with strip + lower.
        q = (mod_search or "").strip().lower()
        if q:
            mods = mods[mods["search_blob"].str.contains(q, na=False, regex=False)]

        if mods.empty:
            st.caption("No modules found")
            st.stop()

        # Summary caption (modules, stages, majors).
        n_modules = int(mods["module_code"].nunique())
        stage_vals = pd.to_numeric(mods["module_stage"], errors="coerce").dropna().astype(int).unique()
        n_stages = int(len(stage_vals))

        st.caption(
            f"Showing {_plural(n_modules, 'module')} "
            f"in {_plural(n_stages, 'stage')} "
            f"from {_plural(1, 'major')}"
        )

        ok = render_col2_stage_module_list(
            mods=mods,
            selected_major_code=st.session_state.selected_major_code,
            activate_major_on_select=True,
        )
        if ok is False:
            st.caption("No staged modules found")

    # Global module search mode: no major selected; search over the full long table.
    else:
        st.header("Module Search")

        mod_search = st.text_input(
            "Search Modules",
            label_visibility="collapsed",
            placeholder=f"Search all {N_MODULES:,} modules",
            key=f"mod_search_box_global_{epoch}",
        )

        if not mod_search.strip():
            st.stop()

        q = mod_search.lower()
        mods = df_mods_by_major[df_mods_by_major["search_blob"].str.contains(q, na=False, regex=False)]

        if mods.empty:
            st.caption("No matches found")
            st.stop()

        n_modules = int(mods["module_code"].nunique())
        stage_vals = pd.to_numeric(mods["module_stage"], errors="coerce").dropna().astype(int).unique()
        n_stages = int(len(stage_vals))

        maj_series = mods["major_code"].fillna("").astype(str).str.strip().str.lower()
        maj_series = maj_series[~maj_series.isin(["", "nan", "none"])]
        n_majors = int(maj_series.nunique())

        st.caption(
            f"Showing {_plural(n_modules, 'module')} "
            f"in {_plural(n_stages, 'stage')} "
            f"from {_plural(n_majors, 'major')}"
        )

        ok = render_col2_stage_module_list(
            mods=mods,
            selected_major_code="__GLOBAL__",
            activate_major_on_select=False,
        )
        if ok is False:
            st.caption("No staged modules found")

# =============================================================================
# COLUMN 3: MODULE DETAILS PANE
# =============================================================================
with col_details:
    if st.session_state.selected_module_code:
        row = module_details_lookup.get(str(st.session_state.selected_module_code))
        if not row:
            st.stop()

        # Primary header: module code and title.
        st.header(f"{row['module_code']} - {row['module_title']}")

        # Subtitle: prefer the explicitly provided subtitle from the selection event.
        subtitle = st.session_state.get("selected_module_subtitle") or ""

        # If no subtitle was provided, synthesize one from module details.
        if not isinstance(subtitle, str) or not subtitle.strip():
            module_type = str(row.get("module_type") or "").capitalize()

            # Credits are formatted as “X Credits” where X is numeric when possible.
            credit_str = ""
            credits = row.get("module_credits")
            if pd.notna(credits):
                try:
                    c = float(credits)
                    credit_str = f"{int(c)} Credits" if c.is_integer() else f"{c} Credits"
                except Exception:
                    credit_str = f"{credits} Credits"

            parts = []

            # Change (5): only include the selected major in the subtitle if this module
            # actually appears in that major.
            maj = st.session_state.get("selected_major_code")
            if isinstance(maj, str) and maj.strip():
                valid_majors = module_to_majors.get(str(row["module_code"]), [])
                if maj in valid_majors:
                    parts.append(maj.strip())

            if module_type.strip():
                parts.append(module_type.strip())

            coord = row.get("module_coordinator_name") or ""
            if isinstance(coord, str) and coord.strip():
                parts.append(coord.strip())

            tri = row.get("module_trimester") or ""
            if isinstance(tri, str) and tri.strip():
                parts.append(tri.strip())

            if isinstance(credit_str, str) and credit_str.strip():
                parts.append(credit_str.strip())

            subtitle = " - ".join(parts) if parts else ""

        if subtitle:
            st.markdown(f"<div class='col3-subtitle'>{subtitle}</div>", unsafe_allow_html=True)

        # Module description (expander).
        description = row.get("module_description", "")
        if isinstance(description, str) and description.strip():
            with st.expander("Module description", expanded=False):
                st.write(description)

        # Eligibility constraints (single expander, multiple sections).
        st.write(" ")
        with st.expander("Eligibility", expanded=False):
            list_cols = [
                "has_prerequisite_modules", "prerequisite_module_for",
                "has_corequisite_modules", "corequisite_module_for",
                "has_incompatible_modules",
                "has_learning_requirement_modules", "learning_requirement_module_for",
            ]
            list_headers = [
                "Prerequisites:",
                "Prerequisite for:",
                "Corequisites:",
                "Corequisite for:",
                "Incompatible with:",
                "Learning requirement:",
                "Learning requirement for:",
            ]

            rendered_any = False
            for col, header in zip(list_cols, list_headers):
                modules = row.get(col)
                if isinstance(modules, (list, tuple, np.ndarray)) and len(modules) > 0:
                    render_module_list_section(
                        module_codes=modules,
                        header_text=header,
                        button_key_prefix=f"{col}_{row['module_code']}",
                    )
                    rendered_any = True

            if not rendered_any:
                st.caption("This module has no listed eligibility constraints.")

        # Similar modules (Top-N, two categories).
        st.write(" ")
        with st.expander("Similar modules", expanded=False):
            render_module_list_section(
                module_codes=row.get("top_n_modules_same_school"),
                header_text="Same school:",
                button_key_prefix=f"top_n_same_school_{row['module_code']}",
                n=5,
            )
            render_module_list_section(
                module_codes=row.get("top_n_modules_different_school"),
                header_text="Different school:",
                button_key_prefix=f"top_n_diff_school_{row['module_code']}",
                n=5,
            )

        # Other majors containing this module (expander).
        st.write(" ")
        module_code = str(row["module_code"])
        majors_for_module = module_to_majors.get(module_code, [])

        current_major = st.session_state.selected_major_code
        if isinstance(current_major, str) and current_major.strip():
            majors_for_module = [m for m in majors_for_module if m != current_major]

        with st.expander("Other majors", expanded=False):
            if not majors_for_module:
                st.caption("This module does not appear in any other majors.")
            else:
                for maj_code in sorted(majors_for_module):
                    meta = major_meta_lookup.get(maj_code, {})
                    programme_title = meta.get("programme_title", "")

                    subtitle = " - ".join(
                        p for p in [
                            meta.get("programme_level", ""),
                            meta.get("programme_award", ""),
                            meta.get("programme_duration", ""),
                            meta.get("programme_attendance", ""),
                        ]
                        if p
                    ) or " "

                    label = f"{maj_code} - {programme_title}\n{subtitle}"

                    if st.button(label, key=f"other_major_{maj_code}", type="secondary"):
                        handle_programme_selection(maj_code, programme_title, subtitle)
                        st.session_state.forced_major_filter = maj_code
                        st.session_state.page_number = 0
                        st.session_state.selected_module_code = None
                        st.rerun()