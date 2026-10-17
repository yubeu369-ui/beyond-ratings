# Reproducible analysis

This folder contains the final Group 24 analysis used in the final report and interactive website.

Run:
```bash
pip install -r requirements.txt
python analysis/final_analysis.py
```

The script looks for `data/matched302.csv`. If it is absent, it reads the same frozen 302-movie sample embedded in the deployed website.

It reproduces:
- RQ1 Pearson correlations, Williams tests for overlapping correlations, and Holm correction.
- RQ2 standardized regression, HC3 robust intervals, and direct coefficient contrasts.
- RQ3 early, later, and total-revenue theater models.
- Breusch-Pagan, residual-shape, VIF, Cook's D, and log-outcome robustness checks.

The code intentionally distinguishes statistical association from causal identification.