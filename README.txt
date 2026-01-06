---

## `README.txt` (plain text, terminal-friendly)

```text
UCD MODULE EXPLORER
==================

An interactive Streamlit application for exploring university programmes (majors)
and their associated modules, including curriculum structure, eligibility
constraints, and module similarity relationships.

The app is designed for fast browsing of programme and module
data, and supports both programme-centric and global module search workflows.


FEATURES
--------
- Programme / major search with pagination
- Staged module listings (core and option modules)
- Global module search across all programmes
- Detailed module view including:
  • Description
  • Eligibility constraints (prerequisites, corequisites, incompatibilities,
    learning requirements)
  • Similar modules
  • Other programmes containing the module


DIRECTORY STRUCTURE
-------------------
ucd_module_explorer/
  app/
    app.py
    data/
      streamlit_major_results.parquet
      streamlit_major_meta.parquet
      streamlit_modules_by_major.parquet
      streamlit_module_details.parquet
  README.md
  README.txt
  LICENSE
  NOTICE
  CITATION.cff
  requirements.txt


RUNNING THE APP
---------------
From the app directory:

  streamlit run app.py

The data directory must be present and populated with the required parquet files.


DATA
----
All data is generated upstream by a preprocessing pipeline and written as
Streamlit-ready parquet artifacts. The app performs no scraping or heavy data
processing at runtime.


CITATION
--------
If you use this software or derived outputs in academic work, please cite:

Barry Smyth. UCD Module Explorer. University College Dublin, 2025.
GitHub: https://github.com/barrysmyth/ucd-module-explorer

A machine-readable citation is provided in CITATION.cff.


LICENSE
-------
Apache License, Version 2.0. See LICENSE and NOTICE for details.


AUTHOR
------
Barry Smyth
University College Dublin
https://github.com/barrysmyth