# Revision notes

## 2026-09-20 — Notebook methodology and reproducibility revision

This change log records the scope of the revised project and its intended validation design. Execution status and measured outcomes belong in the executed notebook and generated reports; these notes do not assert that a run or test suite has passed.

### Data handling

- Add a complete 29-variable dictionary in Markdown and CSV, with separately measured observed values and source limitations.
- Replace deletion of different-ID matching profiles with grouped splitting. Matching predictor profiles stay together across every fit, policy-validation and test boundary; `Response` does not participate in the group key.
- Apply explicit eligibility rules: birth year 1920–1996 inclusive, and income strictly above 0 and below 600,000 or missing. The supplied 2,240 records yield 2,236 eligible records after excluding three implausible birth years and one extreme income; no exact repeated records are present.
- Retain all four records labelled `Absurd` or `YOLO` as `Unknown`; map `Alone` to `Single`. Retain the 24 missing incomes and different-ID matching profiles. The cleaned sample contains 2,034 predictor-profile groups.
- Keep fixed row-quality rules auditable. Move learned imputation, clipping, scaling and supervised feature selection inside training pipelines.
- Use portable project-relative paths and preserve the original notebook and source files as references.

### Model selection and evaluation

- Replace the single exploratory test comparison with five group-aware outer evaluation folds, each containing a separate policy-validation partition and three-fold inner model search.
- Give Logistic Regression, Random Forest, XGBoost and RBF SVM comparable search budgets of eight configurations each. Select by inner-CV ROC-AUC, rather than privileging a model because of historical test performance.
- Use consistent prediction policies for validation and evaluation, including SVM. Do not mix `predict` with a separately thresholded probability rule when comparing threshold metrics.
- Select global and group thresholds on policy-validation predictions. Keep the outer fold unavailable for model, feature and policy decisions.
- Treat previously viewed historical data honestly: nested results assess the revised procedure but do not create a new external or pristine prospective test set.
- Select the final saved candidate using a fixed fit/validation partition. Persist the fitted pipeline and its validated policy without refitting the estimator after its thresholds are chosen.

### Business and fairness interpretation

- Separate fixed top-20% budget targeting, global F1-threshold targeting and fairness-adjusted targeting. Report each policy's own contact volume and metrics.
- Remove retrospective test-profit maximisation from deployment recommendations. If displayed for exploration, identify it as a hindsight optimum.
- Treat contact cost 3 and response value 11 as illustrative assumptions. Remove claims that they prove actual net profit, causal uplift or realised ROI.
- Prespecify the children-at-home fairness grouping instead of selecting a mitigation attribute from test disparities.
- Present the 0.80 selection-rate-ratio threshold as an illustrative benchmark. Explain group support, uncertainty, competing fairness metrics and unavailable sensitive attributes.
- Limit feature explanations and segment descriptions to associations and model behaviour; avoid causal or universal claims.

### Reproducible outputs and documentation

- Use executed metric objects as the source for narrative results, avoiding stale hand-copied thresholds or performance tables.
- Save the raw-input model pipeline and the actual selected policy, including any group-specific cutoffs, rather than saving only classifier weights and an obsolete global threshold.
- Document exploratory segmentation separately and retain its fitted preprocessing when saving its model.
- Provide run instructions, dependencies, reusable source modules and an AI-assistance disclosure.
- Support Python 3.9–3.12 in a new isolated environment with NumPy 1.26.4 pinned. A local NumPy 2.0.2/Accelerate build produced spurious warnings on finite matrix operations; isolated NumPy 1.26.4/OpenBLAS passed the numerical check and full segmentation run without warnings. No global package downgrade is required.

Presentations, GitHub publication, cloud deployment and a demo video are not part of this revision. They should follow a successful, inspected notebook run.
