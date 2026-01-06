# UCD Module Explorer

An interactive Streamlit application for exploring university programmes (majors)
and their associated modules, including curriculum structure, eligibility
constraints, and module similarity relationships.

The app is designed for fast browsing of programme and module
data, and supports both programme-centric and global module search workflows.

---

## Features

- Programme / major search with pagination
- Staged module listings (core vs option modules)
- Global module search across all programmes
- Detailed module view including:
  - Description
  - Eligibility constraints (prerequisites, corequisites, incompatibilities, learning requirements)
  - Similar modules
  - Other programmes containing the module
- Optimised for dense information display with minimal UI friction

---

## Directory Structure

```text
ucd_module_explorer/
├── app/
│   ├── app.py              # Single-file Streamlit application
│   └── data/
│       ├── streamlit_major_results.parquet
│       ├── streamlit_major_meta.parquet
│       ├── streamlit_modules_by_major.parquet
│       └── streamlit_module_details.parquet
├── README.md
├── README.txt
├── LICENSE
├── NOTICE
├── CITATION.cff
└── requirements.txt