"""Fold-local feature engineering and grouped nested model evaluation.

All learned transformations live inside each estimator's sklearn Pipeline.
The outer test folds never enter model selection or policy fitting.
"""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

SPEND = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
PURCHASES = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
CAMPAIGNS = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5']
REF_DATE = '2014-06-30'
SEED = 42


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))


def clean_data(raw):
    """Fixed eligibility rules, no fitted statistics or target-dependent row removal.

    Matching profiles with distinct IDs are retained; profile groups prevent
    their separation across fitting, policy-validation and evaluation folds.
    """
    df = raw.copy()
    log = [('Raw rows', len(df))]
    df = df.drop_duplicates()
    log.append(('Remove exact repeated records', len(df)))
    df = df.loc[df.Year_Birth.between(1920, 1996)].copy()
    log.append(('Eligibility: birth year 1920 to 1996', len(df)))
    df = df.loc[df.Income.isna() | df.Income.between(0, 600000, inclusive='neither')].copy()
    log.append(('Eligibility: positive income below 600000 or missing', len(df)))
    # Rare labels are retained as unknown instead of assuming people are invalid.
    df['Marital_Status'] = df.Marital_Status.replace({'Alone':'Single','Absurd':'Unknown','YOLO':'Unknown'})
    df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer, format='%d-%m-%Y', errors='raise')
    if (df.Dt_Customer > pd.Timestamp(REF_DATE)).any():
        raise ValueError('A customer date exceeds the fixed historical reference date.')
    df = df.reset_index(drop=True)
    feature_cols = [c for c in df if c not in ['ID','Response','Z_CostContact','Z_Revenue']]
    # Target is deliberately excluded from the grouping key, including conflicts.
    keys = pd.util.hash_pandas_object(df[feature_cols], index=False)
    groups = pd.factorize(keys, sort=True)[0]
    log.append(('Retain distinct customer IDs and create profile groups', len(df)))
    return df, groups, pd.DataFrame(log, columns=['step','rows_remaining'])


class CustomerFeatures(TransformerMixin, BaseEstimator):
    """Deterministic raw-to-numeric features, including fixed category indicators."""
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = self.transform(X.iloc[:1]).columns.to_numpy()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        required = SPEND + PURCHASES + CAMPAIGNS + ['Income','Year_Birth','Dt_Customer','Kidhome','Teenhome','Marital_Status','Education','Recency','Complain','NumDealsPurchases','NumWebVisitsMonth']
        missing = sorted(set(required) - set(X.columns))
        if missing: raise ValueError('Missing raw features: ' + ', '.join(missing))
        f = X[['Income','Recency','Complain','NumDealsPurchases','NumWebVisitsMonth'] + SPEND + PURCHASES + CAMPAIGNS].astype(float).copy()
        f['Income_missing'] = X.Income.isna().astype(float)
        f['Age'] = 2014 - X.Year_Birth
        dates = pd.to_datetime(X.Dt_Customer, dayfirst=True)
        f['Tenure_Days'] = (pd.Timestamp(REF_DATE)-dates).dt.days
        f['Children_At_Home'] = X.Kidhome + X.Teenhome
        f['Has_Children_At_Home'] = (f.Children_At_Home > 0).astype(float)
        f['Has_Partner'] = X.Marital_Status.isin(['Married','Together']).astype(float)
        f['Household_Size_Proxy'] = 1 + f.Has_Partner + f.Children_At_Home
        f['Total_Spend'] = X[SPEND].sum(axis=1)
        f['Total_Purchases'] = X[PURCHASES].sum(axis=1)
        # Counts and spend may have mismatched windows; these are proxies.
        denom = f.Total_Purchases.replace(0, np.nan)
        f['Spend_Per_Recorded_Purchase'] = f.Total_Spend / denom
        f['No_Recorded_Purchases'] = (f.Total_Purchases == 0).astype(float)
        f['Deal_Share_Proxy'] = X.NumDealsPurchases / denom
        f['Web_Share'] = X.NumWebPurchases / denom
        f['Wine_Share'] = X.MntWines / f.Total_Spend.replace(0, np.nan)
        f['Meat_Share'] = X.MntMeatProducts / f.Total_Spend.replace(0, np.nan)
        f['Prev_Accepted'] = X[CAMPAIGNS].sum(axis=1)
        f['Spend_to_Income_Proxy'] = f.Total_Spend / X.Income.replace(0, np.nan)
        for cat in ['Basic','2n Cycle','Graduation','Master','PhD']:
            f['Education_' + cat.replace(' ','_')] = (X.Education == cat).astype(float)
        f['Education_Unknown'] = (~X.Education.isin(['Basic','2n Cycle','Graduation','Master','PhD'])).astype(float)
        f['Marital_Unknown'] = (X.Marital_Status == 'Unknown').astype(float)
        return f.replace([np.inf,-np.inf], np.nan).astype(float)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class QuantileCap(TransformerMixin, BaseEstimator):
    """Learn upper 99% caps for explicitly selected skewed numeric features."""
    def __init__(self, quantile=0.99): self.quantile = quantile
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.columns_ = [c for c in ['Income','Spend_Per_Recorded_Purchase','Deal_Share_Proxy','Spend_to_Income_Proxy'] if c in X]
        self.caps_ = X[self.columns_].quantile(self.quantile).to_dict()
        return self
    def transform(self, X):
        z = X.copy()
        for c, cap in self.caps_.items(): z[c] = z[c].clip(upper=cap)
        return z
    def get_feature_names_out(self, input_features=None): return self.feature_names_in_


class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Compute XGBoost class weight using only the data in each fit call."""
    def __init__(self, n_estimators=200, max_depth=2, learning_rate=0.05, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.learning_rate=learning_rate; self.min_child_weight=min_child_weight
        self.subsample=subsample; self.colsample_bytree=colsample_bytree; self.random_state=random_state
    def fit(self, X, y):
        yy=np.asarray(y); self.scale_pos_weight_=float((yy==0).sum()/max((yy==1).sum(),1))
        self.model_=XGBClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,subsample=self.subsample,colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight_,random_state=self.random_state,n_jobs=1,eval_metric='logloss')
        self.model_.fit(X, y); self.classes_=self.model_.classes_; self.n_features_in_=X.shape[1]
        return self
    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)


def model_candidates(seed=SEED):
    models={
        'Logistic Regression': LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=3000,random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=200,class_weight='balanced',n_jobs=1,random_state=seed),
        'XGBoost': BalancedXGBClassifier(random_state=seed),
        'SVM RBF': SVC(kernel='rbf',class_weight='balanced',probability=True,random_state=seed),
    }
    spaces={
        'Logistic Regression': {'clf__C':[0.01,0.1,1,10], 'clf__penalty':['l1','l2']},
        'Random Forest': {'clf__max_depth':[4,8,None], 'clf__min_samples_leaf':[3,8,15], 'clf__max_features':['sqrt',0.7]},
        'XGBoost': {'clf__max_depth':[2,3], 'clf__learning_rate':[0.03,0.08], 'clf__n_estimators':[150,300], 'clf__min_child_weight':[3,7]},
        'SVM RBF': {'clf__C':[0.1,1,10,30], 'clf__gamma':['scale',0.01,0.1]},
    }
    results={}
    for name, model in models.items():
        # Named DataFrames preserve a traceable feature schema through the pipeline.
        pipe=Pipeline([
            ('features',CustomerFeatures()),
            ('imputer',SimpleImputer(strategy='median',keep_empty_features=True).set_output(transform='pandas')),
            ('cap',QuantileCap()),
            ('variance',VarianceThreshold().set_output(transform='pandas')),
            ('select',SelectPercentile(score_func=f_classif,percentile=80).set_output(transform='pandas')),
            ('scale',StandardScaler().set_output(transform='pandas')),
            ('clf',model),
        ])
        results[name]=(pipe,dict(spaces[name],select__percentile=[60,80,100]))
    return results


def score_auc(estimator, X, y):
    """The same probability-ranking API is used for every model and split."""
    return roc_auc_score(y, estimator.predict_proba(X)[:,1])


def group_split(X, y, groups, n_splits=5, seed=42):
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    a,b=next(cv.split(X,y,groups))
    assert set(np.asarray(groups)[a]).isdisjoint(np.asarray(groups)[b])
    return a,b


def tune_models(X,y,groups,seed=42,n_iter=8,inner_folds=3):
    cv=StratifiedGroupKFold(n_splits=inner_folds,shuffle=True,random_state=seed)
    splits=list(cv.split(X,y,groups))
    for a,b in splits: assert set(np.asarray(groups)[a]).isdisjoint(np.asarray(groups)[b])
    fitted={}; records=[]; searches=[]
    for name,(pipe,space) in model_candidates(seed).items():
        search=RandomizedSearchCV(pipe,space,n_iter=n_iter,scoring=score_auc,cv=splits,n_jobs=1,
                                  random_state=seed,error_score='raise',refit=True,return_train_score=False)
        search.fit(X,y)
        fitted[name]=search.best_estimator_
        records.append({'model':name,'inner_cv_auc':search.best_score_,'inner_cv_sd':search.cv_results_['std_test_score'][search.best_index_],
                        'params':search.best_params_})
        for i,p in enumerate(search.cv_results_['params']):
            searches.append({'model':name,'configuration':i,'mean_auc':search.cv_results_['mean_test_score'][i],
                             'sd_auc':search.cv_results_['std_test_score'][i],'params':p})
        print(f'  {name}: inner CV AUC {search.best_score_:.4f}',flush=True)
    table=pd.DataFrame(records).sort_values(['inner_cv_auc','model'],ascending=[False,True])
    return fitted, table, searches


def income_edges(X):
    cuts=np.unique(X.Income.dropna().quantile([.25,.5,.75]).values)
    if len(cuts)!=3: raise ValueError('Cannot form four distinct training-income quartiles.')
    return cuts.tolist()


def top_fraction(scores, fraction=0.2):
    scores=np.asarray(scores); pred=np.zeros(len(scores),dtype=int)
    pred[np.argsort(-scores,kind='stable')[:int(np.ceil(len(scores)*fraction))]]=1
    return pred


def run_nested(df,groups,root,outer_folds=5,inner_folds=3,n_iter=8,seed=42):
    from .fairness import make_sensitive,fit_policy,apply_policy,evaluate_predictions,audit_fairness
    root=Path(root); (root/'reports/tables').mkdir(parents=True,exist_ok=True)
    y=df.Response.astype(int); X=df.drop(columns=['Response'])
    outer=StratifiedGroupKFold(n_splits=outer_folds,shuffle=True,random_state=seed)
    candidates=[]; predictions=[]; policies=[]; searches=[]; audits=[]; split_records=[]
    for fold,(dev,test) in enumerate(outer.split(X,y,groups),1):
        print(f'OUTER FOLD {fold}/{outer_folds}',flush=True)
        fit_local,val_local=group_split(X.iloc[dev],y.iloc[dev],groups[dev],seed=seed+fold)
        fit=dev[fit_local]; val=dev[val_local]
        assert set(groups[fit]).isdisjoint(groups[val]) and set(groups[dev]).isdisjoint(groups[test])
        fitted,rank,search=tune_models(X.iloc[fit],y.iloc[fit],groups[fit],seed+fold,n_iter,inner_folds)
        selected=rank.iloc[0]['model']
        edges=income_edges(X.iloc[fit]); sens_val=make_sensitive(X.iloc[val],edges); sens_test=make_sensitive(X.iloc[test],edges)
        # Every candidate uses validation scores only for its decision rule.
        for name,model in fitted.items():
            p_val=model.predict_proba(X.iloc[val])[:,1]; p=model.predict_proba(X.iloc[test])[:,1]
            policy=fit_policy(y.iloc[val].to_numpy(),p_val,sens_val['children_at_home'].to_numpy())
            pred=apply_policy(p,sens_test['children_at_home'].to_numpy(),policy)
            global_pred=(p>=policy['global_threshold']).astype(int)
            candidates.append({'fold':fold,'model':name,'chosen_by_inner_cv':name==selected,
                               **evaluate_predictions(y.iloc[test].to_numpy(),p,pred),
                               'inner_cv_auc':float(rank.set_index('model').loc[name,'inner_cv_auc'])})
            if name==selected:
                top=top_fraction(p)
                for j,idx in enumerate(test):
                    predictions.append({'row':int(idx),'ID':int(df.ID.iloc[idx]),'group':int(groups[idx]),'fold':fold,'model':name,
                        'y':int(y.iloc[idx]),'score':float(p[j]),'policy_pred':int(pred[j]),'global_pred':int(global_pred[j]),
                        'top20_pred':int(top[j]),**{c:str(sens_test.iloc[j][c]) for c in sens_test.columns}})
                policies.append({'fold':fold,'selected_model':name,'policy':policy,'income_edges':edges,'fit_rows':len(fit),'validation_rows':len(val),'test_rows':len(test)})
                audits.append({'fold':fold,'audit':audit_fairness(y.iloc[test].to_numpy(),pred,sens_test)})
        for r in search: searches.append(dict(r,outer_fold=fold))
        for partition,indices in [('fit',fit),('policy_validation',val),('outer_test',test)]:
            split_records.extend({'fold':fold,'row':int(i),'group':int(groups[i]),'partition':partition} for i in indices)
        print(f'  Frozen inner-CV choice: {selected}',flush=True)
    pred=pd.DataFrame(predictions).sort_values('row').reset_index(drop=True)
    assert len(pred)==len(df) and pred.row.nunique()==len(df)
    candidates=pd.DataFrame(candidates)
    candidates.to_csv(root/'reports/tables/nested_model_comparison.csv',index=False)
    pred.to_csv(root/'reports/tables/nested_predictions.csv',index=False)
    pd.DataFrame(split_records).to_csv(root/'reports/tables/split_manifest.csv',index=False)
    write_json(root/'reports/nested_policies.json',policies)
    write_json(root/'reports/nested_searches.json',searches)
    write_json(root/'reports/nested_fairness_by_fold.json',audits)
    return candidates,pred,policies


def fit_final_candidate(df,groups,root,inner_folds=3,n_iter=8,seed=42):
    from .fairness import make_sensitive,fit_policy,apply_policy,evaluate_predictions
    root=Path(root); (root/'models').mkdir(parents=True,exist_ok=True)
    X=df.drop(columns='Response'); y=df.Response.astype(int)
    fit,val=group_split(X,y,groups,seed=seed+100)
    print('FINAL CANDIDATE: separate fit / policy-validation partition',flush=True)
    fitted,rank,search=tune_models(X.iloc[fit],y.iloc[fit],groups[fit],seed+100,n_iter,inner_folds)
    name=rank.iloc[0]['model']; model=fitted[name]
    edges=income_edges(X.iloc[fit]); sensitive=make_sensitive(X.iloc[val],edges)
    p=model.predict_proba(X.iloc[val])[:,1]
    policy=fit_policy(y.iloc[val].to_numpy(),p,sensitive['children_at_home'].to_numpy())
    selected=model[:-1].get_feature_names_out().tolist()
    versions={m:__import__(m).__version__ for m in ['numpy','pandas','sklearn','scipy','xgboost','joblib']}
    meta={'model':name,'policy':policy,'selected_features':selected,'reference_date':REF_DATE,
          'fit_rows':len(fit),'policy_validation_rows':len(val),'random_state':seed,
          'income_edges':edges,'python':platform.python_version(),'package_versions':versions,
          'validation_role':'Used to fit thresholds; these are not independent test results.',
          'final_estimator_refit_after_policy':False,
          'inference_input':'Eligible raw customer records with the original predictor column names.',
          'raw_data_sha256':hashlib.sha256((root/'data/raw/marketing_campaign.csv').read_bytes()).hexdigest()}
    bundle={'pipeline':model,'policy':policy,'income_edges':edges,'metadata':meta}
    joblib.dump(bundle,root/'models/final_bundle.joblib',compress=3)
    write_json(root/'models/model_config.json',meta)
    rank.assign(params=rank.params.map(lambda v:json.dumps(json_safe(v)))).to_csv(root/'models/model_comparison.csv',index=False)
    write_json(root/'reports/final_candidate_search.json',search)
    pd.DataFrame({'row':np.r_[fit,val],'partition':['fit']*len(fit)+['policy_validation']*len(val)}).to_csv(root/'reports/tables/final_candidate_split.csv',index=False)
    return bundle,rank,fit,val


def predict_bundle(bundle, raw):
    """Apply fixed eligibility/category conventions and the saved targeting policy."""
    from .fairness import make_sensitive,apply_policy
    raw=raw.copy()
    if not raw.Year_Birth.between(1920,1996).all(): raise ValueError('Outside historical age eligibility. Review before scoring.')
    if not (raw.Income.isna() | raw.Income.between(0,600000,inclusive='neither')).all(): raise ValueError('Outside income eligibility.')
    if (pd.to_datetime(raw.Dt_Customer,dayfirst=True)>pd.Timestamp(REF_DATE)).any(): raise ValueError('Customer date exceeds the historical reference date.')
    raw['Marital_Status']=raw.Marital_Status.replace({'Alone':'Single','Absurd':'Unknown','YOLO':'Unknown'})
    scores=bundle['pipeline'].predict_proba(raw)[:,1]
    sensitive=make_sensitive(raw,bundle['income_edges'])
    pred=apply_policy(scores,sensitive['children_at_home'].to_numpy(),bundle['policy'])
    return pd.DataFrame({'score':scores,'contact':pred},index=raw.index)
