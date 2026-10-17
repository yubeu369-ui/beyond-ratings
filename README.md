# G24 Interactive Webpage — Beyond Ratings

Group 24 final data-story website and reproducibility repository.

## Live website
https://yubeu369-ui.github.io/beyond-ratings/

## Reproducible analysis
- `analysis/final_analysis.py` — final Python analysis script
- `analysis/README.md` — run instructions and analysis scope
- `requirements.txt` — Python dependencies

The final analysis reproduces:
- RQ1 raw Pearson correlations and Williams tests for overlapping correlations with Holm correction.
- RQ2 standardized multiple regression, HC3 robust intervals, and direct coefficient contrasts.
- RQ3 early, later, and total-revenue theater models with HC3 robust standard errors.
- Diagnostic and robustness checks including Breusch–Pagan, VIF, Cook's D, and a log-outcome model.

## Main result
The original hypothesis received partial support. Popularity was more strongly associated with domestic revenue than rating in the raw matched sample, while the opening-theater difference versus rating did not pass the planned .05 threshold after Holm correction. After adjustment, the point-estimate ranking changed but direct coefficient gaps were not significant. Opening-week theater exposure showed the most stable positive adjusted relationship across early and later revenue, while remaining vulnerable to expected-demand and distribution confounding.

## Interpretation
This project intentionally distinguishes association from causation. The theater analysis is reported as a stable adjusted association consistent with an audience-access mechanism, not as a proven causal effect.
