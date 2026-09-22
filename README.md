# Customer response prediction and exploratory segmentation

This capstone studies whether historical customer records can help prioritise a marketing campaign while making the trade-offs between response, contact volume and group access explicit. Binary classification of `Response` is the primary task. Customer clustering provides a supporting view of marketing segments.

The revised project addresses evaluation and reporting problems in the original notebook. It keeps matching customer profiles in the same data split, learns preprocessing and feature selection within training folds, chooses decision policies on separate validation data, and separates illustrative business scenarios from demonstrated business outcomes.

## Start here

- [Revised notebook](notebooks/Capstone_Customer_Segmentation_Revised.ipynb): problem framing, data checks, EDA, modelling, segmentation, explanations and fairness analysis.
- [Data dictionary](data/DATA_DICTIONARY.md): all 29 raw variables, observed values, missing counts and interpretation limits.
- [Machine-readable dictionary](data/data_dictionary.csv): the same variable-level reference in CSV format.
- [Change log](CHANGELOG.md): methodological changes from the original notebook.

## Data and project scope

Source: [Customer Personality Analysis on Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data), published by Akash Patel and credited in the data card to Dr. Omar Romero-Hernandez. The data card lists CC0: Public Domain. The supplied file has 2,240 rows and 29 columns. Read it with a tab separator; the `.csv` extension does not indicate its actual delimiter.

The fixed eligibility rules in `src/pipeline.py` retain **2,236 records**: birth year must be between **1920 and 1996 inclusive**, and income must be **strictly greater than 0 and below 600,000, or missing**. Three implausible birth-year records and one extreme income record are excluded. No exact duplicate records are present. `Alone` becomes `Single`; all four records labelled `Absurd` or `YOLO` remain and become `Unknown`. The 24 missing income values are retained for pipeline-local imputation. Different IDs with matching profiles also remain; the eligible records form 2,034 predictor-profile groups for splitting. These documented eligibility assumptions are not source-confirmed corrections or universal customer validity rules.

The data do not establish the company, country, currency, campaign dates or how the sample was assembled. Spending and other activity windows are not timestamped relative to the response campaign. Campaign history is potentially useful, but its availability before the scoring cutoff is an assumption that must be checked before prospective use.

The original notebook and source data are reference material. This revised copy is a separate project. Presentations, cloud deployment and publishing a GitHub repository are outside the current revision.

## Run the project

Use **Python 3.9–3.12**. From this project's root, create and activate a new isolated Python environment, install the supplied dependencies, and execute the notebook:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python execute_notebook.py
```

The requirements pin `numpy==1.26.4`. In this local macOS environment, the existing NumPy 2.0.2 build linked to Accelerate raised numerical warnings even for small finite matrix operations. An isolated NumPy 1.26.4/OpenBLAS environment passed the numerical check and complete segmentation run without those warnings. The pinned environment provides the reproducible workaround; a global NumPy downgrade is unnecessary.

Place the supplied tab-separated dataset at `data/raw/marketing_campaign.csv` if it is not already included. The runner executes `notebooks/Capstone_Customer_Segmentation_Revised.ipynb` and writes its outputs back to that notebook. Run from the project root so relative paths resolve consistently. The grouped model searches require more computation than the original single split; allow the run to finish before interpreting partially written results.

The delivered notebook was executed in Python 3.9.6 with the pinned dependencies. All 18 analysis code cells completed without errors or warning streams, and all 17 unit tests passed. Saved model-policy and segmentation roundtrip checks passed. See `reports/verification.json`. These checks validate implementation, not future campaign performance.

```text
capstone-revised/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── execute_notebook.py
├── notebooks/
│   └── Capstone_Customer_Segmentation_Revised.ipynb
├── src/                     # reusable cleaning, modelling and analysis code
├── data/
│   ├── raw/marketing_campaign.csv
│   ├── processed/
│   ├── DATA_DICTIONARY.md
│   └── data_dictionary.csv
├── models/                  # fitted pipeline, decision policy and supporting artefacts
├── reports/                 # generated metrics, tables and figures
│   └── figures/
└── tests/                   # relevant checks, where supplied
```

## Evaluation design

1. Apply the documented birth-year and income eligibility rules and remove exact repeated records if present. Map rare relationship labels to `Unknown`, preserving their records. Retain distinct customer IDs even when their measured profiles match. Build a grouping key from the available profile while excluding `ID`, `Response` and the two constant `Z_` fields; every split keeps matching profiles together. A shared profile is a leakage-control heuristic, not proof of shared identity.
2. Use five stratified, group-aware outer folds. Each outer test fold is held aside while every model and policy decision for that fold is made.
3. Split each outer-training partition approximately 80:20 into model-fit data and policy-validation data, again preserving groups and approximately preserving class balance.
4. Tune Logistic Regression, Random Forest, XGBoost and an RBF SVM on model-fit data. Give each candidate eight configurations and three group-aware inner CV folds. Learned imputation, clipping, scaling and feature selection are refitted within those inner training folds. Select the candidate by mean inner-CV ROC-AUC only.
5. Use policy-validation predictions to set the global F1 threshold and the prespecified children-at-home fairness adjustment. Neither the outer test labels nor their subgroup disparities select the algorithm, features, cutoffs or adjustment strength.
6. Evaluate the frozen candidate and policies on that fold's outer test data. Aggregate these out-of-fold results to estimate the staged model-selection procedure. Top-20% targeting is a separate, predeclared budget policy; any retrospective best-profit point is not a deployable performance estimate.
7. Build the final saved candidate with an independent fixed fit/validation partition of the cleaned dataset. Select and fit the model using its fit portion; calibrate the decision policy on its validation portion. Persist that exact fitted pipeline and policy **without refitting after threshold selection**. The final validation result is a development result, not an independent test of the saved model.

The nested outer results evaluate a procedure across held-out folds. They are not a new external cohort, and they are not a pristine prospective test: this historical dataset was already explored in the original notebook. Earlier exploratory results may have influenced this revision's design. Future validation should use a later campaign or a new dataset with a verified feature cutoff.

ROC-AUC drives model selection. Precision-recall performance, precision and capture at the fixed 20% budget, threshold metrics, contact volume and group results provide complementary decision evidence. They should be reported together rather than choosing whichever test metric happens to favour one candidate. The latest executed output determines the winning algorithm; XGBoost is not predetermined as the winner.

## Segmentation and explanation

Segmentation is exploratory and is not used as a classifier input. Its preprocessing, candidate cluster counts, silhouette results and business interpretation belong together. A preference for three interpretable groups over two better-separated groups must be presented as a practical trade-off rather than proof of natural customer types. Saved clustering artefacts should include their fitted preprocessing and feature order.

Feature importance and local or global response plots explain model behaviour, not causal effects. Correlated and mathematically related inputs can make partial-dependence scenarios unrealistic. Neither a feature's high ranking nor an attractive segment profile establishes that changing it would improve response.

## Business scenario and fairness

The constants `Z_CostContact = 3` and `Z_Revenue = 11` lack confirmed economic definitions in the linked data card. Scenario calculations therefore use **an assumed cost of 3 per contact and an assumed value of 11 per observed responder**. Currency, contribution margin, other costs and incremental response are unknown. Report the result as an illustrative scenario value; it is not established net profit, realised ROI or causal campaign uplift. A no-contact control or suitable experiment is needed for causal claims.

The children-at-home grouping is specified before evaluation. Group-specific thresholds trade off selection rates, missed responders and contact costs; the chosen adjustment must be learned on policy-validation data and frozen before testing. A disparate-impact ratio of 0.80 is an illustrative screening benchmark, not a universal fairness definition, legal test or guarantee. Report group support, rates and uncertainty, along with demographic parity and equalised-odds metrics. Improving one metric may worsen another.

Audit age, income, education and household attributes where available. The absence of race and gender fields prevents auditing those dimensions. Observed historical response differences cannot distinguish preferences from earlier targeting practices, unequal access or omitted context.

## Saved artefacts and reuse

Keep the fitted raw-input preprocessing and classifier together, along with feature names, fixed reference dates, grouping definitions, random seeds, package versions and the actual decision policy. A fairness-adjusted recommendation requires its group thresholds and adjustment strength in addition to model weights. Keep fixed-budget targeting separate from threshold targeting; they generally contact different people and have different metrics.

Only load serialized Python model artefacts from a trusted source. For a future campaign, verify the schema and prediction-time availability of every input, then monitor contact volume, response rates, drift and group outcomes. Any refit needs a new validation step for its thresholds; old cutoffs need not transfer unchanged.

## AI assistance disclosure

Generative AI assisted with the original-notebook audit, code restructuring and explanatory documentation. The analysis remains conventional statistical and machine-learning modelling; an LLM does not generate customer labels or campaign-response predictions. The project author should inspect the executed outputs, verify interpretations and disclose this assistance in the final submission. No demo video or presentation is included in this notebook revision.

## Verification

Run `python -m unittest discover -s tests -v`. The checks cover split isolation, training-only preprocessing, frozen decision policies, subgroup metrics and saved-artifact reproducibility.
