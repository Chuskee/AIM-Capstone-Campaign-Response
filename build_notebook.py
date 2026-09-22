"""Create the editable notebook source; execute_notebook.py performs the run."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
md('''# Customer Campaign Response and Supporting Segmentation
## Revised capstone notebook

**Primary task:** binary prediction of acceptance of the recorded last campaign. **Supporting task:** exploratory customer segmentation.

This notebook covers rubric Steps 1–5 and prepares reproducible local artifacts for Step 7. It revises the original analysis through fold-specific preprocessing, group-aware nested validation, comparable tuning, validation-only targeting policies, complete fairness audits and saved inference bundles. Technical and business presentations are separate deliverables.

**Dataset:** [Customer Personality Analysis on Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data), credited by the data card to Dr. Omar Romero-Hernandez. The snapshot contains historical customer data. Country, currency and exact campaign timing are not established here.

**Evaluation disclosure:** the original notebook already examined this dataset and its holdout. The revised procedure uses nested grouped cross-validation for a more disciplined internal estimate. No partition is represented as newly collected external data. New-campaign or external validation remains necessary.

**How to use:** extract the complete project and run `python execute_notebook.py` from its root after installing `requirements.txt`. Keep `src/`, `data/` and the notebook together. All tables and findings below are generated from the same run.''')
md('''## Step 1 Problem understanding and measurable objectives

The business scenario is a marketing team with limited outreach capacity. Historical labels identify customers who accepted the last recorded campaign. Predictive ranking could concentrate outreach on likely responders, while a fairness audit checks whether the targeting policy distributes opportunities unevenly.

| Item | Definition |
|---|---|
| Prediction target | `Response`: 1 accepted the last campaign, 0 did not |
| Model-selection metric | Mean inner-validation ROC-AUC; fixed before evaluation |
| Additional model metrics | Average precision (AP), F1, responder recall, precision, top-20% precision/capture |
| Outreach policy | Validation-fitted F1 threshold, then a predefined children-at-home disparity adjustment |
| Fairness objective | Smallest adjustment reaching selection-rate ratio 0.80 on policy-validation data; this is a project assumption, not a guarantee of fairness |
| Comparison budget | Fixed top 20% by model score, evaluated as an alternative policy |
| Business measures | Contacts, response concentration, cost per observed responder, and an illustrative value proxy |
| Illustrative economics | Value 11 per recorded responder and contact cost 3, in unspecified units; unverified contribution margin |
| Provisional technical targets | Outer-fold mean AUC ≥0.80; pooled top-20% response rate ≥2× overall prevalence; positive illustrative policy value |
| Adoption gate | Review uncertainty, all-group disparities, privacy and feature timing; validate prospectively before use |

The acceptance targets are proposed for this revision, not stakeholder-approved commitments. The project predicts historical response, not causal campaign uplift. A controlled experiment would be needed to establish incremental business value.''')
md('''## Reproducibility and configuration

The input file is tab-separated despite its `.csv` extension. Fixed cleaning rules precede splitting. No income imputation, percentile clipping, supervised feature selection or classifier scaling is fitted on evaluation data.

Five outer folds evaluate the complete selection procedure. Inside each outer development fold, a grouped 80/20 split separates model fitting from policy validation. Four candidate algorithms each receive eight sampled configurations with three inner grouped folds. The outer fold is used only after the model and policy have been frozen.''')
code('''from pathlib import Path
import os, sys, json, hashlib, platform
ROOT = Path.cwd()
if not (ROOT / "data/raw/marketing_campaign.csv").exists():
    ROOT = ROOT.parent
assert (ROOT / "data/raw/marketing_campaign.csv").exists(), "Open this notebook from the extracted project."
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
for folder in ["data/processed", "models", "reports/figures", "reports/tables"]:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay, roc_auc_score, average_precision_score
from src.pipeline import (clean_data, CustomerFeatures, run_nested, fit_final_candidate, predict_bundle,
                          write_json, json_safe, SPEND, PURCHASES, REF_DATE)
from src.fairness import make_sensitive, apply_policy, evaluate_predictions, audit_fairness
RANDOM_STATE = 42
CONFIG = {"outer_folds": 5, "inner_folds": 3, "n_iter": 8, "seed": RANDOM_STATE}
write_json(ROOT / "run_config.json", CONFIG)
pd.set_option("display.max_columns", 18)
pd.set_option("display.max_rows", 60)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def save_plot(name):
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures" / (name + ".png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
versions = {name: __import__(name).__version__ for name in ["numpy", "pandas", "sklearn", "scipy", "xgboost", "shap"]}
print("Python", platform.python_version(), "| package versions", versions)
print("Reproducible configuration:", CONFIG)''')
md('''## Step 2 Data collection and understanding

The source data card and local dictionary distinguish observed values from allowed-domain assumptions. Income and spending have unspecified currency; spending totals refer to two years, but the purchase-count observation windows are not verified. Those limitations matter when engineering ratios and interpreting business value.''')
code('''raw_path = ROOT / "data/raw/marketing_campaign.csv"
df_raw = pd.read_csv(raw_path, sep="\\t")
assert df_raw.shape[1] == 29
print("Raw shape:", df_raw.shape)
print("Dataset SHA256:", hashlib.sha256(raw_path.read_bytes()).hexdigest())
overview = pd.DataFrame({"type": df_raw.dtypes.astype(str), "missing": df_raw.isna().sum(),
                         "missing_pct": 100 * df_raw.isna().mean(), "unique": df_raw.nunique()})
display(overview.round(2))
display(df_raw.head(3))
print("Exact duplicates:", df_raw.duplicated().sum())
print("Matching profiles ignoring ID and Response:", df_raw.drop(columns=["ID", "Response"]).duplicated().sum())
display(df_raw.describe().T.round(2))''')
code('''dictionary = pd.read_csv(ROOT / "data/data_dictionary.csv")
assert len(dictionary) == 29
# The Markdown version is also readable without running this notebook.
display(dictionary)
display(Markdown("Full definitions and source caveats: [data dictionary](../data/DATA_DICTIONARY.md)."))''')
md('''## Step 3 Cleaning and applied exploratory analysis

Cleaning uses explicitly documented historical eligibility assumptions: birth years 1920–1996 and positive income below 600,000 when present. The three very old birth years and income 666,666 are treated as suspicious and excluded; this is an analytical assumption rather than proof that the records are incorrect. Missing income stays missing until each training pipeline fits its median.

`Alone` is normalised to `Single`. Rare `Absurd`/`YOLO` marital labels become `Unknown`; their customers remain. Distinct customer IDs with matching profiles remain too, because equal profiles do not prove the same person. A profile key excludes both ID and Response and forces all matching profiles into the same partition, including conflicting responses. Fixed filtering restricts the population to which conclusions apply.''')
code('''df, groups, cleaning_log = clean_data(df_raw)
display(cleaning_log)
print(f"Eligible customers: {len(df):,}; distinct profile groups: {len(np.unique(groups)):,}")
print(f"Income missing before fold-local imputation: {df.Income.isna().sum()}")
print("Retained rare marital labels:", df.Marital_Status.value_counts().to_dict())
df.assign(Profile_Group=groups).to_csv(ROOT / "data/processed/customers_eligible.csv", index=False)
cleaning_log.to_csv(ROOT / "reports/tables/cleaning_log.csv", index=False)
eda = CustomerFeatures().fit_transform(df.drop(columns="Response"))
assert not np.isinf(eda.to_numpy()).any()
print("Engineered feature count before fold-specific variance filtering/selection:", eda.shape[1])''')
md('''### Feature engineering decisions

| Feature family | Reason and limitation |
|---|---|
| Age and tenure | Fixed historical reference 2014-06-30 makes the calculation reproducible; it is not a verified campaign date |
| Total spending and channel purchases | Summaries of observed commercial activity |
| Product and channel shares | Relative preferences within the same source variable family |
| Spend per recorded purchase and spend-to-income | Labelled proxies because observation windows differ or are unknown |
| Zero-purchase and missing-income flags | Preserve an indication of unavailable or inconsistent inputs |
| Children at home and household-size proxy | Household context, subject to explicit fairness audit; not proof of parenthood |
| Previous accepted campaigns | Candidate historical behaviour, assuming campaigns 1–5 precede the last campaign; timing remains unverified |
| Education indicators | Fixed one-hot categories avoid imposing an unsupported ordinal spacing |

The classifier pipeline learns medians and selected numeric caps within each fit. A variance filter removes constant predictors, and an ANOVA filter selects 60%, 80% or 100% of the remaining predictors through inner validation. Its univariate ranking may miss interactions; 100% retention is therefore a candidate. Scaling stays inside the pipeline. The tree models do not require scaling, but the monotone scaling allows a common audited feature path.''')
code('''prevalence = df.Response.mean()
fig, ax = plt.subplots(figsize=(6, 3.5))
counts = df.Response.value_counts().sort_index()
ax.bar(["No response", "Response"], counts.values, color=["#96a2ad", "#247b91"])
ax.set_ylabel("Customers"); ax.set_title(f"Recorded campaign acceptance: {prevalence:.1%}")
for i,v in enumerate(counts): ax.text(i,v+10,str(v),ha="center")
save_plot("01_response_balance")
display(Markdown(f"**Observed imbalance:** {int(df.Response.sum()):,} of {len(df):,} customers responded ({prevalence:.1%}). An always-no classifier obtains {(1-prevalence):.1%} accuracy while capturing no responders. This motivates ranking, precision and recall metrics."))''')
code('''columns = ["Income", "Age", "Recency", "Tenure_Days", "Total_Spend", "Total_Purchases", "Spend_Per_Recorded_Purchase", "NumWebVisitsMonth"]
fig, axes = plt.subplots(2,4,figsize=(14,6))
for ax,col in zip(axes.ravel(),columns):
    ax.hist(eda[col].dropna(),bins=30,color="#247b91",alpha=.85)
    ax.set_title(col.replace("_", " "),fontsize=10)
    ax.set_ylabel("Customers")
save_plot("02_distributions")
display(eda[columns].describe().T.round(2))
display(Markdown("**Interpretation:** income and spending include long upper tails. Ratios based on small recorded purchase counts can be extreme. Training-only caps limit their influence; zero denominators stay missing and are imputed inside the training pipeline. These plots describe the reused snapshot and are not independent validation evidence."))''')
code('''# EDA-only grouping; inferential audit quartiles are later fitted separately per training fold.
eda_income_cuts = df.Income.dropna().quantile([.25,.5,.75]).tolist()
eda_groups = make_sensitive(df, eda_income_cuts)
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
for ax,col in zip(axes,["income_quartile","children_at_home","education"]):
    table=df.assign(group=eda_groups[col]).groupby("group").Response.agg(["mean","size"])
    ax.bar(table.index,table["mean"],color="#247b91");ax.axhline(prevalence,color="#9b4f34",ls="--")
    ax.set_title(col.replace("_"," "));ax.tick_params(axis="x",rotation=35)
    ax.set_ylabel("Historical response rate")
    for i,row in enumerate(table.itertuples()): ax.text(i,row.mean+.007,f"n={row.size}",ha="center",fontsize=8)
save_plot("03_group_response")
corr_features=["Income","Age","Recency","Tenure_Days","Total_Spend","Total_Purchases","Children_At_Home","Prev_Accepted","NumWebVisitsMonth"]
corr=eda[corr_features].assign(Response=df.Response).corr()
fig,ax=plt.subplots(figsize=(8,6))
im=ax.imshow(corr,vmin=-1,vmax=1,cmap="RdBu_r")
ax.set_xticks(range(len(corr)));ax.set_xticklabels(corr.columns,rotation=70,ha="right")
ax.set_yticks(range(len(corr)));ax.set_yticklabels(corr.index)
fig.colorbar(im,ax=ax,label="Pearson correlation")
ax.set_title("Associations and redundant predictors")
save_plot("04_correlations")
display(corr.Response.drop("Response").sort_values(key=abs,ascending=False).to_frame("correlation").round(3))
display(Markdown("**Interpretation:** group response differences and correlated spend features warrant a fairness audit and cautious explanation. Correlations identify associations; they do not establish marketing effects, individual preferences or legitimate grounds to exclude a group."))''')
md('''## Step 4 Model implementation and selection

The four algorithms provide complementary baselines: a regularised linear classifier, bagged trees, boosted trees and an RBF support-vector classifier. All receive the same search budget and the same inner group folds. Class weights are fitted from each model-fitting partition. The same `predict_proba` output is used for ranking and validation/test threshold policies, avoiding SVM decision-rule mismatches.

**Frozen decision sequence in every outer fold**
1. Reserve the outer evaluation fold with no matching profile groups in development.
2. Split development into model-fitting and policy-validation groups.
3. Tune each algorithm on model-fitting data only, including preprocessing and feature selection. Select the largest mean inner ROC-AUC; deterministic name order breaks exact ties.
4. Fit the global F1 threshold on policy validation. Apply the prespecified children-at-home adjustment, selecting the smallest alpha in 0, 0.05, …, 1 that attains the illustrative 0.80 selection ratio there. If no alpha succeeds, retain the global policy and explicitly flag that failure.
5. Evaluate the frozen models and the selected procedure on the outer fold. Do not change the algorithm, attribute, budget or alpha using that fold.

Nested evaluation estimates this complete selection procedure. It does not make the historical dataset externally representative. AUC and other outer results below are reporting evidence, not a second rule for choosing the final candidate. Fitting eight configurations is a bounded comparison, not exhaustive optimisation.''')
code('''# Main training cell: all learned preprocessing and selection are inside the folds.
comparison, outer_predictions, outer_policies = run_nested(df, groups, ROOT, **CONFIG)
write_json(ROOT / "reports/evaluation_contract.json", {
    "estimate": "Grouped nested internal validation of the complete model and targeting-policy selection procedure",
    "historical_data_previously_explored": True,
    "final_candidate_selection": "Highest inner CV ROC-AUC on a separate final fit partition",
    "targeting_attribute": "children_at_home, prespecified",
    "top_fraction_alternative": 0.20,
    "outer_test_used_for_decision_selection": False,
    "hyperparameter_configurations_per_algorithm": CONFIG["n_iter"]})
metrics_cols=["roc_auc","average_precision","f1","precision","recall","contact_fraction"]
comparison_summary=comparison.groupby("model")[metrics_cols].agg(["mean","std"])
display(comparison_summary.round(3))
comparison_summary.to_csv(ROOT / "reports/tables/model_outer_summary.csv")
selected_folds=comparison.loc[comparison.chosen_by_inner_cv]
display(selected_folds[["fold","model","inner_cv_auc"]+metrics_cols].round(3))
policy_diagnostics=pd.DataFrame([{"fold":p["fold"],"model":p["selected_model"],"policy_status":p["policy"]["status"],
    "alpha":p["policy"]["alpha"],"validation_selection_ratio":p["policy"]["validation_metrics"]["selection_rate_ratio"],
    "fit_rows":p["fit_rows"],"validation_rows":p["validation_rows"],"test_rows":p["test_rows"]} for p in outer_policies])
display(policy_diagnostics)
policy_diagnostics.to_csv(ROOT / "reports/tables/nested_policy_diagnostics.csv",index=False)''')
code('''fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,metric,label in [(axes[0],"roc_auc","ROC-AUC"),(axes[1],"average_precision","Average precision")]:
    stats=comparison.groupby("model")[metric].agg(["mean","std"])
    ax.barh(stats.index,stats["mean"],xerr=stats["std"],color="#247b91",capsize=4)
    ax.set_xlabel(label);ax.set_xlim(0,1);ax.set_title("Outer-fold mean and fold SD")
save_plot("05_model_comparison")
selection_counts=selected_folds.model.value_counts()
display(Markdown("**Model-selection evidence:** " + "; ".join(f"{name} selected in {n}/{CONFIG['outer_folds']} outer development folds" for name,n in selection_counts.items()) + ". The error bars show fold variability, not confidence intervals. A small average difference alone is insufficient to establish a definitive winner."))''')
md('''### Complete-policy performance and fixed-budget alternative

Each customer below is evaluated once by a model and policy that did not fit that customer's profile group. Pooled metrics describe this procedure across five separately trained models. Income-quartile group labels use each fold's training-derived cutpoints. A mean of fold AUCs and a pooled AUC are distinct summaries and are both labelled. Different fold models can have different uncalibrated score scales, so mean outer-fold AUC is the primary ranking estimate; pooled AUC and AP are supplementary summaries.

The top-20% comparison fixes the contact budget before evaluation. There is no test-label search for a retrospectively profitable campaign size. “Observed-label value” applies assumed economics to historical outcomes and is not measured incremental profit.''')
code('''y_o=outer_predictions.y.to_numpy(); p_o=outer_predictions.score.to_numpy()
policy_columns={"Selected fairness-adjusted policy":"policy_pred","Global F1 policy":"global_pred","Fixed top 20%":"top20_pred"}
policy_metrics={name:evaluate_predictions(y_o,p_o,outer_predictions[col].to_numpy()) for name,col in policy_columns.items()}
policy_metrics["Contact everyone"]=evaluate_predictions(y_o,p_o,np.ones(len(y_o),dtype=int))
policy_metrics["Contact nobody"]=evaluate_predictions(y_o,p_o,np.zeros(len(y_o),dtype=int))
policy_table=pd.DataFrame(policy_metrics).T
responder_counts=pd.to_numeric(policy_table.tp)
policy_table["cost_per_observed_responder"]=pd.to_numeric(policy_table.assumed_contact_spend) / responder_counts.where(responder_counts.ne(0))
policy_table["response_concentration_lift"]=policy_table.precision/prevalence
show_cols=["selected_n","tp","precision","recall","f1","contact_fraction","observed_label_profit_proxy","observed_label_roi_proxy","cost_per_observed_responder","response_concentration_lift"]
display(policy_table[show_cols].round(3))
policy_table.to_csv(ROOT / "reports/tables/policy_comparison.csv")
write_json(ROOT / "reports/policy_comparison.json",policy_metrics)
fig,axes=plt.subplots(1,3,figsize=(15,4))
fpr,tpr,_=roc_curve(y_o,p_o);axes[0].plot(fpr,tpr,color="#247b91");axes[0].plot([0,1],[0,1],ls="--",color="gray")
axes[0].set(xlabel="False-positive rate",ylabel="True-positive rate",title=f"Pooled nested ROC-AUC {roc_auc_score(y_o,p_o):.3f}")
pr,re,_=precision_recall_curve(y_o,p_o);axes[1].plot(re,pr,color="#247b91");axes[1].axhline(prevalence,ls="--",color="gray")
axes[1].set(xlabel="Recall",ylabel="Precision",title=f"Pooled AP {average_precision_score(y_o,p_o):.3f}")
ConfusionMatrixDisplay.from_predictions(y_o,outer_predictions.policy_pred,display_labels=["No","Yes"],ax=axes[2],colorbar=False,cmap="Blues")
axes[2].grid(False);axes[2].set_title("Selected policy, outer predictions")
save_plot("06_selected_policy")''')
code('''# Cluster bootstrap: matching customer profiles resample together.
# Conditional on already fitted fold models; does not rerun model selection.
rng=np.random.default_rng(RANDOM_STATE)
blocks=[g.index.to_numpy() for _,g in outer_predictions.groupby("group")]
boot=[]
for _ in range(400):
    idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
    b=outer_predictions.iloc[idx]
    met=evaluate_predictions(b.y,b.score,b.policy_pred)
    rates=b.groupby("children_at_home").policy_pred.mean()
    boot.append({"roc_auc":met["roc_auc"],"average_precision":met["average_precision"],"f1":met["f1"],
                 "recall":met["recall"],"precision":met["precision"],
                 "children_selection_ratio":rates.min()/rates.max() if rates.max()>0 else np.nan})
intervals=pd.DataFrame(boot).quantile([.025,.975]).T
intervals.columns=["conditional_95pct_lower","conditional_95pct_upper"]
display(intervals.round(3))
intervals.to_csv(ROOT / "reports/tables/conditional_bootstrap_intervals.csv")
display(Markdown("**Uncertainty limitation:** these profile-cluster bootstrap intervals hold fold-trained models fixed. They describe sensitivity to the evaluated customer sample, not all model-training uncertainty. Outer folds also share some training records. Small differences in models or fairness metrics should therefore be interpreted cautiously."))''')
md('''## Step 5 Ethical AI and bias auditing

The relevant decision is access to a marketing offer. Both unwanted contact and missed beneficial offers can matter. Age, income, education, children living at home and partnership status are descriptive audit axes; the dataset contains no gender or race information. Missing income and unknown marital labels retain explicit audit groups.

The mitigation axis is **children at home**, specified before evaluation because the earlier project identified household access as a concern. The numerical 0.80 target is an illustrative design choice, not evidence of legal compliance or fairness. It may trade off against predictive performance, other groups and equalised odds.

Demographic parity gap measures the maximum minus minimum selection rate. The selection ratio is the minimum divided by maximum rate. Equalised-odds gap is the larger of TPR/FPR gaps. Undefined rates remain missing rather than being treated as zero. Group counts and Wilson intervals are displayed; those intervals assume independent Bernoulli outcomes within groups, so the profile-cluster bootstrap is the more suitable overall sensitivity check for repeated profiles.''')
code('''sensitive_columns=["age_band","income_quartile","education","children_at_home","household"]
audits={}
for name,col in policy_columns.items():
    audits[name]=audit_fairness(y_o,outer_predictions[col].to_numpy(),outer_predictions[sensitive_columns])
    pd.DataFrame(audits[name]["groups"]).to_csv(ROOT / "reports/tables" / (col+"_fairness_groups.csv"),index=False)
    pd.DataFrame(audits[name]["summary"]).to_csv(ROOT / "reports/tables" / (col+"_fairness_summary.csv"),index=False)
write_json(ROOT / "reports/fairness_audits.json",audits)
nested_audit=pd.DataFrame(audits["Selected fairness-adjusted policy"]["summary"])
display(Markdown("**All-group audit of the nested selection procedure.** Undefined equalised-odds values indicate that at least one included group lacks positive or negative outcomes."))
display(nested_audit.round(3))
display(pd.DataFrame(audits["Selected fairness-adjusted policy"]["groups"]).round(3))
axis_ratios=nested_audit.set_index("attribute").disparate_impact_ratio
display(Markdown(f"**Remaining disparities:** the pooled children-at-home selection ratio is {axis_ratios['children_at_home']:.3f}. "
    + "Other ratios are " + "; ".join(f"{name.replace('_',' ')} {ratio:.3f}" for name,ratio in axis_ratios.items() if name!="children_at_home")
    + ". The adjustment targets one axis and does not establish fairness overall. Interpret groups with very few customers with particular caution."))''')
code('''fair_groups=pd.DataFrame(audits["Selected fairness-adjusted policy"]["groups"])
print("Audit table columns:",list(fair_groups.columns))
# Direct group-rate chart uses the actual final decisions, independent of field aliases.
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,col in zip(axes,["children_at_home","income_quartile","age_band"]):
    t=outer_predictions.groupby(col).agg(selection=("policy_pred","mean"),actual_response=("y","mean"),n=("y","size"))
    t[["selection","actual_response"]].plot.bar(ax=ax,color=["#247b91","#a8b2b9"],rot=30)
    ax.set_title(col.replace("_"," "));ax.set_xlabel("");ax.set_ylabel("Rate")
    ax.legend(fontsize=8)
save_plot("07_final_policy_groups")
display(Markdown("**Audit interpretation:** validation attainment of the 0.80 target does not ensure attainment on outer folds or for other attributes. Compare group denominators, uncertainty and equalised odds before recommending use. Historical response disparities may reflect opportunity, sampling or measurement differences; they are not proof of intrinsic preferences."))''')
md('''### Business sensitivity and consistent policy reporting

The same fairness-adjusted decisions supply every value calculation in this section. The contact cost and response value are scenario assumptions, not verified current prices or margins. Values at different assumptions are a sensitivity analysis; the scenario with the highest test value is not used to change the selected policy.''')
code('''contacted=int(outer_predictions.policy_pred.sum())
responders=int(((outer_predictions.policy_pred==1)&(outer_predictions.y==1)).sum())
sensitivity=pd.DataFrame([{ "value_per_response":v,"cost_per_contact":c,
                           "observed_label_value":responders*v-contacted*c}
                         for v in [6,11,16] for c in [1,3,5]])
display(sensitivity.pivot(index="value_per_response",columns="cost_per_contact",values="observed_label_value"))
sensitivity.to_csv(ROOT / "reports/tables/value_sensitivity.csv",index=False)
mean_auc=float(selected_folds.roc_auc.mean())
top_lift=float(policy_table.loc["Fixed top 20%","response_concentration_lift"])
reference_value=float(policy_table.loc["Selected fairness-adjusted policy","observed_label_profit_proxy"])
gates=pd.DataFrame({"criterion":["Mean outer AUC >= 0.80","Top-20% response concentration >= 2x","Positive illustrative selected-policy value"],
                    "observed":[mean_auc,top_lift,reference_value],
                    "met":[mean_auc>=.80,top_lift>=2,reference_value>0]})
display(gates)
display(Markdown("These technical checks do not authorise deployment. A prospective pilot must verify data availability, consent, offer eligibility, actual costs and group outcomes."))''')
md('''## Final candidate and reproducible persistence

The final candidate is chosen with a separate deterministic group split of the whole eligible dataset. Its fitting partition supplies the same inner-CV algorithm selection; its validation partition supplies only threshold fitting. Outer results do not determine the final algorithm. The model is **not refitted after choosing thresholds**, because refitting could change the score scale.

This means the saved candidate leaves its policy-validation customers out of model fitting. It sacrifices some fitting data to preserve a consistent estimator-policy pairing. Its validation metrics are not an independent performance estimate; the nested procedure above provides the internal evaluation. Save and load `final_bundle.joblib`, which contains the whole pipeline, sensitive-group definitions through code, thresholds and metadata.''')
code('''bundle,final_ranking,final_fit_idx,final_validation_idx = fit_final_candidate(
    df,groups,ROOT,inner_folds=CONFIG["inner_folds"],n_iter=CONFIG["n_iter"],seed=RANDOM_STATE)
display(final_ranking[["model","inner_cv_auc","inner_cv_sd","params"]])
print("Saved final candidate:",bundle["metadata"]["model"])
print("Global threshold:",round(bundle["policy"]["global_threshold"],4))
print("Fairness alpha:",bundle["policy"]["alpha"])
print("Policy fitting status:",bundle["policy"]["status"])
print("Final group thresholds:",bundle["policy"]["group_thresholds"])
print("Selected features:",bundle["metadata"]["selected_features"])
# A fresh load must reproduce the exact scores and contact decisions.
import joblib
loaded=joblib.load(ROOT / "models/final_bundle.joblib")
example=df.drop(columns="Response").iloc[final_validation_idx[:12]]
before=predict_bundle(bundle,example);after=predict_bundle(loaded,example)
np.testing.assert_allclose(before.score,after.score,rtol=0,atol=0)
np.testing.assert_array_equal(before.contact,after.contact)
print("Saved-pipeline score and policy roundtrip: PASS")
display(after)
candidate_validation=df.drop(columns="Response").iloc[final_validation_idx]
candidate_output=predict_bundle(bundle,candidate_validation)
candidate_sensitive=make_sensitive(candidate_validation,bundle["income_edges"])
candidate_audit=audit_fairness(df.Response.iloc[final_validation_idx].to_numpy(),candidate_output.contact.to_numpy(),candidate_sensitive)
write_json(ROOT / "reports/final_candidate_validation_audit.json",candidate_audit)
display(Markdown("**Saved candidate audit on its policy-validation partition.** These records fitted its thresholds, so this table is a development diagnostic, not independent test evidence."))
display(pd.DataFrame(candidate_audit["summary"]).round(3))''')
md('''### Feature selection and model explainability

The feature-selection report shows the final candidate's fitted filter and retained variables. It is descriptive of that fitted model; selection can differ in other folds.

SHAP explains model output, not causal drivers of customer behaviour. Tree/linear explanations may use a raw margin or log-odds scale, stated in the plot notes. PDP/ICE changes raw Recency and raw web visits and reruns the full pipeline, so deterministic derived features remain consistent. Correlation with other unchanged raw inputs may still produce unusual records. No claim is made that the plotted sample rules out all heterogeneous subgroups.''')
code('''from src.explainability import explain_candidate
explanation=explain_candidate(bundle,df.drop(columns="Response").iloc[final_fit_idx],
                              df.drop(columns="Response").iloc[final_validation_idx],ROOT,RANDOM_STATE)
print(json.dumps(json_safe(explanation),indent=2))
for filename in ["explain_shap.png","explain_pdp_ice.png"]:
    display(Image(filename=str(ROOT / "reports/figures" / filename),width=1000))
display(pd.read_csv(ROOT / "reports/tables/final_feature_selection.csv"))
display(pd.read_csv(ROOT / "reports/tables/final_shap_importance.csv").head(10))''')
md('''## Supporting customer segmentation and PCA

This track groups the whole eligible snapshot without using Response as a clustering feature. Median imputation, selected log transforms and scaling precede Euclidean clustering. K-Means k=2–6 and Ward hierarchical clustering k=2–4 are compared with silhouette, Davies–Bouldin and Calinski–Harabasz scores. The reusable K-Means solution selects its k by the highest silhouette within that family. The best result across families is also disclosed; choosing a reusable centroid model does not establish universal superiority.

PCA describes variance and provides a two-dimensional visualisation. It is not part of the classifier validation path. Response rates are joined to completed clusters only as descriptive profiles, not used to choose the clustering solution. The k=3 alternative and seed stability make the earlier three-segment story testable rather than assumed.''')
code('''from src.segmentation import run_segmentation
segmentation=run_segmentation(df,ROOT,RANDOM_STATE)
print(json.dumps(json_safe(segmentation),indent=2))
write_json(ROOT / "reports/segmentation_summary.json",segmentation)
for key in ["evaluation","profiles"]:
    display(pd.read_csv(ROOT / segmentation["files"][key]))
for key in ["comparison_figure","pca_figure","profiles_figure"]:
    display(Image(filename=str(ROOT / segmentation["files"][key]),width=1050))''')
md('''## Critical limitations and responsible use

- **Timing and leakage:** campaign order and pre-campaign availability of spend/recency/history cannot be verified from this snapshot. The grouped pipeline prevents split-related leakage but cannot fix unknown feature timestamps. Use an as-of campaign feature table and a time-based evaluation before operational use.
- **Historical reuse:** this dataset has informed earlier analysis. Nested validation is an internal correction, not an independent external confirmation. New campaigns may have different base rates, offers and customers.
- **Eligibility and repeated profiles:** retained distinct IDs may represent different customers or repeated records. Grouped splits prevent matching profiles from crossing folds, but population weighting and data lineage remain uncertain.
- **Imbalance and score calibration:** class weighting can affect probability calibration. Outputs are treated as bounded ranking scores; threshold policies are fitted on validation data. A probability-based financial decision would require a separate calibration assessment.
- **Explainability:** correlated features split attribution; feature rankings, SHAP and PDP are associations in a fitted model. Feature availability and domain review matter more than an appealing explanation plot.
- **Fairness:** no gender/race data; coarse age/income groups and small subgroup counts limit conclusions. Selection parity is one objective among several. Missing metrics reflect insufficient denominators, not successful fairness.
- **Privacy and communication:** keep identifiers out of predictors, minimise data access and avoid presenting historical segments as fixed identities. A human should review campaign eligibility, opt-outs and potential harm.
- **Business evidence:** historical response concentration is not causal uplift. The 3/11 value calculation and sensitivity grid are illustrative. A randomised pilot should measure incremental response, actual costs, complaints and group access.

Potential mitigations for a prospective pilot include accurate campaign-time data, calibrated scores, improving offer relevance for underserved groups, and transparent contact constraints. Monitor every campaign by group and pause a policy when material harm or data drift appears. These are recommendations; this notebook does not deploy a service.''')
code('''# Findings use canonical run objects; no manually copied model percentages.
selected_result=policy_metrics["Selected fairness-adjusted policy"]
conclusion=f"""## Findings from this run

The grouped nested procedure achieved mean outer ROC-AUC **{mean_auc:.3f}** and pooled average precision **{selected_result['average_precision']:.3f}**. Its fairness-adjusted decisions contacted **{selected_result['selected_n']:,} of {len(df):,}** evaluated customers, with precision **{selected_result['precision']:.1%}**, recall **{selected_result['recall']:.1%}** and F1 **{selected_result['f1']:.3f}**.

Under the illustrative value of 11 and cost of 3, the exact same decisions have observed-label value **{selected_result['observed_label_profit_proxy']:,.0f}** unspecified units. This is a historical scenario calculation, not verified incremental profit.

The saved candidate is **{bundle['metadata']['model']}**, selected by inner CV on its fitting partition. Its global threshold is **{bundle['policy']['global_threshold']:.3f}** and its validation-selected fairness adjustment is **{bundle['policy']['alpha']:.2f}**. The full bundle stores the exact group thresholds. Its own policy-validation results must not be presented as independent test performance.

The descriptive K-Means analysis selects **k={segmentation['selected_k']}**, with silhouette **{segmentation['selected_silhouette']:.3f}**. Two PCA axes explain **{segmentation['pca']['first_two_explained_variance']:.1%}** of variance. These internal clustering results describe a historical snapshot rather than guaranteeing stable real-world customer types.

The subgroup tables above determine where disparities remain. A validated outreach pilot is the next business evidence step. The technical and business presentations should use these generated results and retain their limitations.

The pooled children-at-home selection ratio is **{axis_ratios['children_at_home']:.3f}**. Other group disparities remain, and the smallest household group contains only four records. Numerical attainment of the illustrative target on one axis is insufficient to declare the policy fair.
"""
display(Markdown(conclusion))
(ROOT / "reports/run_findings.md").write_text(conclusion)
summary={"configuration":CONFIG,"eligible_rows":len(df),"profile_groups":len(np.unique(groups)),
         "selected_procedure_outer_mean_auc":mean_auc,"pooled_policy_metrics":selected_result,
         "final_candidate_model":bundle['metadata']['model'],"final_policy":bundle['policy'],
         "selected_segmentation_k":segmentation['selected_k'],"gates":gates.to_dict(orient="records"),
         "limitations":["Historical dataset previously explored","No external or prospective validation","Unknown feature timing","Illustrative economics only"]}
write_json(ROOT / "reports/run_summary.json",summary)
print("Canonical results, fairness tables, model bundle and segmentation artifacts saved.")''')
md('''## Submission and reproducibility checklist

| Requirement | Evidence in this package |
|---|---|
| Problem statement, task and metrics | Step 1 |
| Dataset overview and dictionary | Step 2 and `data/DATA_DICTIONARY.md` |
| Cleaning, EDA and features | Step 3 and saved figures |
| Feature selection and dimensionality reduction | Fold-specific ANOVA selector; descriptive PCA |
| Multiple models and tuning | Four candidates with matched configuration budgets |
| Reproducibility | `src/`, pinned requirements, configuration, split manifests, saved complete bundles |
| Explainability and fairness | SHAP, coherent raw-feature PDP/ICE, all-group audits and validation-selected mitigation |
| Final report evidence | Executed notebook and generated `reports/run_findings.md` |
| Presentations and public GitHub URL | Separate deliverables, not created by this notebook |

**Generative AI disclosure:** AI assistance revised code, methodology explanations and documentation. Numeric findings come from executed code. The learner should review, understand and be able to explain the methods and limitations before submission. No additional marks or performance claims are assumed.

### References

1. [Kaggle Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data) — source data card and feature meanings.
2. [Scikit-learn data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage-during-pre-processing) — fit preprocessing and selection only within training partitions.
3. [Scikit-learn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) — stratification with disjoint groups.
4. [Scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) — predicted labels and probability thresholds can differ.
5. [Fairlearn fairness metrics](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html) — definitions, context and limits of numerical parity benchmarks.
6. [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) — output scales and feature-dependence assumptions.
7. [Scikit-learn PDP and ICE](https://scikit-learn.org/stable/modules/partial_dependence.html) — interpretation and dependence caveats.
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}})
path=ROOT/'notebooks/Capstone_Customer_Segmentation_Revised.ipynb'
nbf.write(nb,path)
print(f'Created {path} with {len(cells)} cells')
