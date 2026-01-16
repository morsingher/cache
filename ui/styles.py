import altair as alt
import streamlit as st

def apply_custom_css():
    st.markdown(
        """
<style>
/* Secondary buttons */
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background-color: #ecebe3 !important;
}

/* Glide-data-grid (Streamlit dataframe) CSS custom properties */
:root {
    --gdg-bg-cell: #f4f3ed !important;
    --gdg-bg-header: #ecebe3 !important;
    --gdg-bg-header-has-focus: #e8e7dd !important;
    --gdg-bg-header-hovered: #e8e7dd !important;
    --gdg-accent-color: #bb5a38 !important;
    --gdg-accent-light: #ecebe3 !important;
    --gdg-bg-bubble: #ecebe3 !important;
    --gdg-bg-bubble-selected: #e8e7dd !important;
    --gdg-bg-search-result: #ecebe3 !important;
}

/* Additional table styling */
.stDataFrame,
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizableContainer"] {
    --gdg-bg-cell: #f4f3ed !important;
    --gdg-bg-header: #ecebe3 !important;
    --gdg-bg-header-has-focus: #e8e7dd !important;
    --gdg-bg-header-hovered: #e8e7dd !important;
}

/* NOTE: Do NOT globally center headings.
   We only center the custom app title in `_render_title()` via inline HTML styles.
   Leaving headings left-aligned fixes inconsistent section-header alignment. */

/* Reduce space between title (h1) and subtitle (h2) */
[data-testid="stHeading"] h1,
h1[data-testid="stHeading"] {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
[data-testid="stHeading"] h2 {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Style the custom title h1 */
h1[data-testid="stHeading"] {
    font-size: 52px !important;
    font-weight: 600 !important;
}

/* Hide anchor links on all headers */
[data-testid="stHeading"] a,
h1 a[href^="#"],
h2 a[href^="#"],
h3 a[href^="#"],
h4 a[href^="#"] {
    display: none !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

# Register pastel color theme for Altair charts
@alt.theme.register("pastel", enable=True)
def _pastel_theme():
    return {
        "config": {
            "range": {
                "category": [
                    "#AEC6CF",  # Pastel Blue
                    "#FFB7C5",  # Pastel Pink
                    "#B39EB5",  # Pastel Purple
                    "#77DD77",  # Pastel Green
                    "#FDFD96",  # Pastel Yellow
                    "#FFB347",  # Pastel Orange
                    "#CFCFC4",  # Pastel Grey
                    "#F49AC2",  # Pastel Magenta
                    "#B19CD9",  # Pastel Lavender
                    "#FF6961",  # Pastel Red
                ]
            }
        }
    }
