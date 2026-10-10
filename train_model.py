import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

def run_pipeline():
    dataset_path = 'synthetic_loan_dataset (1).csv'
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join('data', 'loan_dataset.csv')
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Target variable
    target_col = 'loan_status'
    id_col = 'loan_id'
    
    X = df.drop(columns=[id_col, target_col])
    # Target encoding: Approved -> 1, Rejected -> 0
    # The user specifies: 'Approved' (positive class), 'Rejected' (negative class)
    y = (df[target_col].str.strip() == 'Approved').astype(int)
    
    print(f"Class distribution: Approved={y.sum()} ({y.mean()*100:.2f}%), Rejected={(1-y).sum()} ({(1-y.mean())*100:.2f}%)")

    cat_cols = ['education', 'self_employed']
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
            ]), cat_cols)
        ]
    )

    # 1. Stratified Train-Test split (80/20) with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

    # 2. Secondary evaluation: Compare reference models
    print("\n--- Running Secondary Model Comparison on 5-Fold Stratified CV (Training set only) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    comparison_models = {
        'Logistic Regression': Pipeline([
            ('prep', ColumnTransformer([
                ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
                ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))]), cat_cols)
            ])),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Decision Tree': Pipeline([
            ('prep', preprocessor),
            ('clf', DecisionTreeClassifier(random_state=42, max_depth=6))
        ]),
        'Random Forest': Pipeline([
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(random_state=42, n_estimators=100))
        ]),
        'Extra Trees': Pipeline([
            ('prep', preprocessor),
            ('clf', ExtraTreesClassifier(random_state=42, n_estimators=100))
        ]),
        'Gradient Boosting (Baseline)': Pipeline([
            ('prep', preprocessor),
            ('clf', GradientBoostingClassifier(random_state=42))
        ])
    }

    comparison_results = []
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    for name, model_pipe in comparison_models.items():
        scores = cross_validate(model_pipe, X_train, y_train, cv=cv, scoring=scoring)
        res = {
            'Model': name,
            'CV Accuracy': float(scores['test_accuracy'].mean()),
            'CV Accuracy Std': float(scores['test_accuracy'].std()),
            'CV Precision': float(scores['test_precision'].mean()),
            'CV Recall': float(scores['test_recall'].mean()),
            'CV F1': float(scores['test_f1'].mean()),
            'CV ROC-AUC': float(scores['test_roc_auc'].mean())
        }
        comparison_results.append(res)
        print(f"{name:28s} | Acc: {res['CV Accuracy']:.4f} | F1: {res['CV F1']:.4f} | ROC-AUC: {res['CV ROC-AUC']:.4f}")

    comp_df = pd.DataFrame(comparison_results).sort_values(by='CV ROC-AUC', ascending=False)
    comp_df.to_csv('reports/model_comparison.csv', index=False)
    print("Saved reports/model_comparison.csv")

    # 3. Hyperparameter tuning for Gradient Boosting using Training data only
    print("\n--- Tuning Gradient Boosting Hyperparameters on Training Set ---")
    gb_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        'clf__n_estimators': [50, 100, 150],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__max_depth': [3, 4, 5],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        gb_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_gb = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    print(f"Best CV ROC-AUC: {best_cv_score:.4f}")
    print(f"Best Parameters: {best_params}")

    # 4. Out-of-fold validation predictions to select decision threshold
    print("\n--- Selecting Decision Threshold on Out-Of-Fold Validation Predictions ---")
    oof_probs = np.zeros(len(X_train))
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_model = Pipeline([
            ('prep', preprocessor),
            ('clf', GradientBoostingClassifier(
                n_estimators=best_params['clf__n_estimators'],
                learning_rate=best_params['clf__learning_rate'],
                max_depth=best_params['clf__max_depth'],
                min_samples_leaf=best_params['clf__min_samples_leaf'],
                subsample=best_params['clf__subsample'],
                random_state=42
            ))
        ])
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]

    # Evaluate thresholds
    thresholds = np.linspace(0.2, 0.8, 61)
    best_thresh = 0.50
    best_f1 = 0.0
    thresh_records = []

    for t in thresholds:
        pred_t = (oof_probs >= t).astype(int)
        f1_t = f1_score(y_train, pred_t)
        prec_t = precision_score(y_train, pred_t, zero_division=0)
        rec_t = recall_score(y_train, pred_t)
        acc_t = accuracy_score(y_train, pred_t)
        tn, fp, fn, tp = confusion_matrix(y_train, pred_t).ravel()
        thresh_records.append({
            'threshold': round(float(t), 3),
            'f1': float(f1_t),
            'precision': float(prec_t),
            'recall': float(rec_t),
            'accuracy': float(acc_t),
            'false_approvals': int(fp),
            'false_rejections': int(fn)
        })
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = round(float(t), 3)

    print(f"Selected decision threshold based on max F1 on validation folds: {best_thresh} (Val F1: {best_f1:.4f})")

    # 5. Fit best model on all training data
    best_gb.fit(X_train, y_train)

    # 6. Evaluate on untouched Test Set
    print("\n--- Evaluating Final Gradient Boosting Model on Untouched Test Set ---")
    y_test_probs = best_gb.predict_proba(X_test)[:, 1]
    y_test_pred_default = best_gb.predict(X_test)
    y_test_pred_thresh = (y_test_probs >= best_thresh).astype(int)

    test_acc = accuracy_score(y_test, y_test_pred_thresh)
    test_prec = precision_score(y_test, y_test_pred_thresh)
    test_rec = recall_score(y_test, y_test_pred_thresh)
    test_f1 = f1_score(y_test, y_test_pred_thresh)
    test_roc_auc = roc_auc_score(y_test, y_test_probs)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_thresh).ravel()
    # Positive class is Approved (1), Negative class is Rejected (0)
    # True Approval: tp (actually approved, predicted approved)
    # True Rejection: tn (actually rejected, predicted rejected)
    # False Approval: fp (actually rejected, predicted approved)
    # False Rejection: fn (actually approved, predicted rejected)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP(False Approvals)={fp}, FN(False Rejections)={fn}, TP={tp}")

    model_evaluation = {
        'model_name': 'GradientBoostingClassifier',
        'hyperparameters': {k.replace('clf__', ''): v for k, v in best_params.items()},
        'positive_class': 'Approved',
        'negative_class': 'Rejected',
        'selected_threshold': best_thresh,
        'threshold_selection_criterion': 'Maximizing F1-score on 5-fold stratified cross-validation out-of-fold predictions',
        'cross_validation_results': {
            'mean_cv_roc_auc': float(best_cv_score),
            'mean_cv_f1': float(best_f1)
        },
        'test_metrics': {
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1_score': float(test_f1),
            'roc_auc': float(test_roc_auc),
            'confusion_matrix': {
                'true_rejections_tn': int(tn),
                'false_approvals_fp': int(fp),
                'false_rejections_fn': int(fn),
                'true_approvals_tp': int(tp)
            },
            'error_counts': {
                'false_approvals_count': int(fp),
                'false_rejections_count': int(fn),
                'total_errors': int(fp + fn)
            }
        },
        'trade_off_analysis': {
            'false_approvals_implication': 'Lending to an applicant who would default or fail criteria, leading to financial loss for the institution.',
            'false_rejections_implication': 'Denying credit to a creditworthy borrower, causing lost interest revenue and poor customer experience.',
            'threshold_recommendation': f'Threshold {best_thresh} strikes an empirical balance on cross-validation.'
        }
    }

    with open('reports/model_evaluation.json', 'w') as f:
        json.dump(model_evaluation, f, indent=4)
    print("Saved reports/model_evaluation.json")

    # 7. Feature Importance & Explainability (Global Permutation & Global SHAP)
    print("\n--- Computing Global Permutation Importance and SHAP Importance ---")
    
    # Get transformed feature names
    prep_fitted = best_gb.named_steps['prep']
    clf_fitted = best_gb.named_steps['clf']

    # Transformed feature names
    cat_encoder = prep_fitted.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    print(f"Transformed feature names ({len(all_feature_names)}): {all_feature_names}")

    # Permutation importance on test set using best_gb pipeline directly
    perm_res = permutation_importance(
        best_gb, X_test, y_test,
        n_repeats=10, random_state=42, scoring='roc_auc'
    )

    perm_importance_df = pd.DataFrame({
        'feature': X.columns,
        'permutation_importance_mean': perm_res.importances_mean,
        'permutation_importance_std': perm_res.importances_std
    }).sort_values(by='permutation_importance_mean', ascending=False)

    # SHAP Explainability
    X_train_trans = prep_fitted.transform(X_train)
    X_test_trans = prep_fitted.transform(X_test)
    tree_explainer = shap.TreeExplainer(clf_fitted)

    # Held-out SHAP values describe the trained model's behaviour on test rows.
    test_shap_explanation = tree_explainer(X_test_trans)
    test_shap_array = np.asarray(test_shap_explanation.values)
    if test_shap_array.ndim == 3:
        test_output = 1 if test_shap_array.shape[-1] > 1 else 0
        test_shap_array = test_shap_array[:, :, test_output]
    if test_shap_array.ndim != 2:
        raise ValueError(f"Unexpected test SHAP output shape: {test_shap_array.shape}")
    mean_abs_shap = np.abs(test_shap_array).mean(axis=0)

    # Select the ablation feature on training data only. The held-out test set remains
    # reserved for the final comparison rather than selecting which feature to remove.
    train_sample_count = min(1000, len(X_train_trans))
    train_shap_sample = X_train_trans[:train_sample_count]
    train_shap_explanation = tree_explainer(train_shap_sample)
    train_shap_array = np.asarray(train_shap_explanation.values)
    if train_shap_array.ndim == 3:
        train_output = 1 if train_shap_array.shape[-1] > 1 else 0
        train_shap_array = train_shap_array[:, :, train_output]
    if train_shap_array.ndim != 2:
        raise ValueError(f"Unexpected training SHAP output shape: {train_shap_array.shape}")
    mean_abs_train_shap = np.abs(train_shap_array).mean(axis=0)

    # Map transformed feature names to the original input feature groups.
    shap_to_orig = {}
    for transformed_feature in all_feature_names:
        matched = None
        for original_feature in X.columns:
            if transformed_feature == original_feature or transformed_feature.startswith(original_feature + "_"):
                matched = original_feature
                break
        shap_to_orig[transformed_feature] = matched or transformed_feature

    test_shap_features_df = pd.DataFrame({
        "transformed_feature": all_feature_names,
        "mean_abs_shap": mean_abs_shap,
    })
    test_shap_features_df["original_feature"] = test_shap_features_df["transformed_feature"].map(shap_to_orig)
    orig_shap = (
        test_shap_features_df.groupby("original_feature")["mean_abs_shap"]
        .sum().reset_index()
    )

    train_shap_features_df = pd.DataFrame({
        "transformed_feature": all_feature_names,
        "mean_abs_shap_train": mean_abs_train_shap,
    })
    train_shap_features_df["original_feature"] = train_shap_features_df["transformed_feature"].map(shap_to_orig)
    orig_shap_train = (
        train_shap_features_df.groupby("original_feature")["mean_abs_shap_train"]
        .sum().reset_index()
        .sort_values("mean_abs_shap_train", ascending=False)
    )
    ablation_feature = str(orig_shap_train.iloc[0]["original_feature"])

    feature_imp_combined = pd.merge(
        perm_importance_df,
        orig_shap.rename(columns={"original_feature": "feature"}),
        on="feature", how="left"
    ).sort_values(by="mean_abs_shap", ascending=False)

    feature_imp_combined.to_csv("reports/feature_importance.csv", index=False)
    print("Saved reports/feature_importance.csv:")
    print(feature_imp_combined)

    # Global ranking used for the dashboard is based on held-out mean absolute SHAP.
    most_influential_feature = str(feature_imp_combined.iloc[0]["feature"])
    print(f"Most influential held-out SHAP feature: {most_influential_feature}")
    print(
        f"Feature selected for training-only ablation: {ablation_feature} "
        f"(SHAP sample rows: {train_sample_count})"
    )

    # 8. Feature-Removal Experiment (Model A: All Features vs Model B: Without Most Influential Feature)
    print("\n--- Running Feature-Removal Experiment ---")
    X_train_b = X_train.drop(columns=[ablation_feature])
    X_test_b = X_test.drop(columns=[ablation_feature])

    cat_cols_b = [c for c in cat_cols if c != ablation_feature]
    num_cols_b = [c for c in num_cols if c != ablation_feature]

    preprocessor_b = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols_b),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
            ]), cat_cols_b)
        ]
    )

    model_b = Pipeline([
        ('prep', preprocessor_b),
        ('clf', GradientBoostingClassifier(
            n_estimators=best_params['clf__n_estimators'],
            learning_rate=best_params['clf__learning_rate'],
            max_depth=best_params['clf__max_depth'],
            min_samples_leaf=best_params['clf__min_samples_leaf'],
            subsample=best_params['clf__subsample'],
            random_state=42
        ))
    ])

    # Model B CV
    scores_b = cross_validate(model_b, X_train_b, y_train, cv=cv, scoring=scoring)
    model_b.fit(X_train_b, y_train)

    y_test_probs_b = model_b.predict_proba(X_test_b)[:, 1]
    y_test_pred_b = (y_test_probs_b >= best_thresh).astype(int)

    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_test_pred_b).ravel()

    experiment_comparison = {
        'model_a_all_features': {
            'features_used': list(X.columns),
            'cv_roc_auc_mean': float(best_cv_score),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_f1': float(test_f1),
            'test_roc_auc': float(test_roc_auc),
            'false_approvals': int(fp),
            'false_rejections': int(fn)
        },
        'model_b_without_dominant_feature': {
            'removed_feature': ablation_feature,
            'features_used': list(X_train_b.columns),
            'cv_roc_auc_mean': float(scores_b['test_roc_auc'].mean()),
            'test_accuracy': float(accuracy_score(y_test, y_test_pred_b)),
            'test_precision': float(precision_score(y_test, y_test_pred_b)),
            'test_recall': float(recall_score(y_test, y_test_pred_b)),
            'test_f1': float(f1_score(y_test, y_test_pred_b)),
            'test_roc_auc': float(roc_auc_score(y_test, y_test_probs_b)),
            'false_approvals': int(fp_b),
            'false_rejections': int(fn_b)
        },
        'performance_drop': {
            'accuracy_delta': float(accuracy_score(y_test, y_test_pred_b) - test_acc),
            'f1_delta': float(f1_score(y_test, y_test_pred_b) - test_f1),
            'roc_auc_delta': float(roc_auc_score(y_test, y_test_probs_b) - test_roc_auc)
        }
    }
    print("Feature Removal Experiment Results:")
    print(f"Model A (All features) Test ROC-AUC: {test_roc_auc:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"Model B (No {ablation_feature}) Test ROC-AUC: {roc_auc_score(y_test, y_test_probs_b):.4f}, Acc: {accuracy_score(y_test, y_test_pred_b):.4f}, F1: {f1_score(y_test, y_test_pred_b):.4f}")

    # 9. Stability Across Folds
    print("\n--- Checking Feature Importance Stability Across Folds ---")
    fold_rankings = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        f_clf = Pipeline([
            ('prep', preprocessor),
            ('clf', GradientBoostingClassifier(
                n_estimators=best_params['clf__n_estimators'],
                learning_rate=best_params['clf__learning_rate'],
                max_depth=best_params['clf__max_depth'],
                min_samples_leaf=best_params['clf__min_samples_leaf'],
                subsample=best_params['clf__subsample'],
                random_state=42
            ))
        ])
        f_clf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        # Tree feature importances
        clf_step = f_clf.named_steps['clf']
        f_importances = clf_step.feature_importances_
        # Map back to original features
        f_dict = {}
        for feat_name, imp in zip(all_feature_names, f_importances):
            orig = shap_to_orig[feat_name]
            f_dict[orig] = f_dict.get(orig, 0.0) + imp
        sorted_feats = sorted(f_dict.items(), key=lambda x: x[1], reverse=True)
        top_3 = [x[0] for x in sorted_feats[:3]]
        fold_rankings.append({
            'fold': fold_i + 1,
            'top_feature': sorted_feats[0][0],
            'top_feature_weight': float(sorted_feats[0][1]),
            'top_3_features': top_3
        })
        print(f"Fold {fold_i+1}: Rank 1={sorted_feats[0][0]} ({sorted_feats[0][1]*100:.1f}%), Top 3={top_3}")

    # Check correlations among features
    corr_matrix = X[num_cols].corr()
    high_corrs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c_val = corr_matrix.iloc[i, j]
            if abs(c_val) > 0.5:
                high_corrs.append({
                    'feature_1': num_cols[i],
                    'feature_2': num_cols[j],
                    'correlation': float(c_val)
                })

    # Produce examples from observed data instead of making unsupported fixed claims.
    correlation_examples = ", ".join(
        f"{pair['feature_1']} / {pair['feature_2']} ({pair['correlation']:.2f})"
        for pair in high_corrs[:3]
    )
    stable_top_count = sum(
        fold["top_feature"] == most_influential_feature for fold in fold_rankings
    )
    auc_without_top = float(roc_auc_score(y_test, y_test_probs_b))
    accuracy_without_top = float(accuracy_score(y_test, y_test_pred_b))

    # Compile dependency analysis report
    dependency_analysis = {
        'most_influential_feature': most_influential_feature,
        'mean_absolute_shap_contribution': float(orig_shap[orig_shap['original_feature'] == most_influential_feature]['mean_abs_shap'].values[0]),
        'feature_removal_experiment': experiment_comparison,
        'stability_across_folds': fold_rankings,
        'high_feature_correlations': high_corrs,
        'ablation_feature_selection': {
            'feature_removed': ablation_feature,
            'method': 'Highest mean absolute SHAP contribution on a sample of training rows',
            'training_sample_rows': int(train_sample_count),
            'held_out_test_set_used_for_selection': False
        },
        'answers_to_research_questions': {
            'q1_most_influential_feature': (
                f"'{most_influential_feature}' ranks first by mean absolute SHAP on the held-out test set. "
                "Permutation importance is also reported and can rank features differently because it measures "
                "the performance change after shuffling each original feature."
            ),
            'q2_measured_contribution': f"Mean absolute SHAP value is {float(orig_shap[orig_shap['original_feature'] == most_influential_feature]['mean_abs_shap'].values[0]):.4f} in log-odds contribution.",
            'q3_performance_decline_on_removal': (
                f"Removing '{ablation_feature}' (selected using training SHAP values) changed held-out ROC-AUC from {test_roc_auc:.4f} "
                f"to {auc_without_top:.4f} (change: {experiment_comparison['performance_drop']['roc_auc_delta']:+.4f}; "
                "a negative change is a decline). Accuracy changed by "
                f"{experiment_comparison['performance_drop']['accuracy_delta']:+.4f}."
            ),
            'q4_do_other_features_provide_value': (
                f"Without '{ablation_feature}', held-out ROC-AUC was {auc_without_top:.4f}, "
                f"compared with {test_roc_auc:.4f} for the full model. This measures remaining predictive signal "
                "on this particular split; it does not prove the target is independent of the removed feature."
            ),
            'q5_stability_across_folds': (
                f"In {stable_top_count} of {len(fold_rankings)} training folds, the top feature by tree impurity "
                f"importance matched '{most_influential_feature}', which was selected by global mean absolute SHAP. "
                "These are different importance measures and should not be treated as identical."
            ),
            'q6_multicollinearity_and_redundancy': (
                f"Found {len(high_corrs)} numeric feature pair(s) with absolute Pearson correlation above 0.50. "
                f"Examples from this dataset: {correlation_examples if correlation_examples else 'none'}."
            ),
            'q7_synthetic_rule_vs_complex_pattern': (
                "The exact synthetic label-generation rule cannot be established reliably from the model results alone. "
                "Inspect the original data-generation process if available. High test scores can reflect patterns built "
                "into synthetic labels and do not establish performance on real lending data."
            ),
            'q8_limitations': "The data is synthetic/simulated; findings reflect the simulation generating process rather than actual institutional underwriting policies. Real lending decisions incorporate credit history depth, debt service ratios, verified employment, and subjective underwriter review."
        }
    }

    with open('reports/dependency_analysis.json', 'w') as f:
        json.dump(dependency_analysis, f, indent=4)
    print("Saved reports/dependency_analysis.json")

    # 10. Save Model Pipeline and Metadata
    print("\n--- Saving Artifacts ---")
    joblib.dump(best_gb, 'models/gradient_boosting_pipeline.pkl')
    print("Saved models/gradient_boosting_pipeline.pkl")

    model_metadata = {
        'model_name': 'GradientBoostingClassifier Pipeline',
        'pipeline_steps': ['prep (ColumnTransformer with SimpleImputer & OneHotEncoder)', 'clf (GradientBoostingClassifier)'],
        'feature_schema': {
            'features': list(X.columns),
            'numerical_features': num_cols,
            'categorical_features': cat_cols,
            'transformed_features': all_feature_names,
            'categorical_options': {col: list(X[col].unique()) for col in cat_cols}
        },
        'target_mapping': {
            '0': 'Rejected',
            '1': 'Approved'
        },
        'selected_threshold': best_thresh,
        'base_expected_value': float(tree_explainer.expected_value[0] if isinstance(tree_explainer.expected_value, (list, np.ndarray)) else tree_explainer.expected_value),
        'metrics_summary': {
            'cv_roc_auc': float(best_cv_score),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_f1': float(test_f1),
            'test_roc_auc': float(test_roc_auc)
        },
        'most_influential_feature': most_influential_feature,
        'training_dataset': dataset_path,
        'synthetic_disclaimer': 'This model was trained on a synthetic dataset for educational and explainability demonstration. It does not represent actual credit decisions made by regulated financial institutions.'
    }

    with open('models/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=4)
    print("Saved models/model_metadata.json")
    print("\n=== Training & Evaluation Complete! ===")

if __name__ == '__main__':
    run_pipeline()
