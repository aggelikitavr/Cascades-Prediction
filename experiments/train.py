import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.feature_extraction import extract_features

def train_at_k(cascades, k=10, growth_factor=2, model_type='logistic'):
    """
    Train a model to predict if a cascade will double in size (growth_factor=2)
    after observing the first k nodes.
    """
    
    # Keep cascades that have at least k nodes for observation
    eligible_cascades = [c for c in cascades if c.size() >= k]

    if len(eligible_cascades) < 50:
        return None

    X = []
    y = []

    for c in eligible_cascades:
        # Extract features strictly using information up to node k
        features = extract_features(c, until=k)
        
        # Label 1 if cascade size eventually reaches growth_factor * k
        label = 1 if c.size() >= growth_factor * k else 0

        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        print(f"Only one class present for k={k}. Skipping.")
        return None

    # Select the classifier
    if model_type == 'logistic': 
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        clf = SVC(kernel='linear', probability=True, random_state=42)
    else:
        raise ValueError("Unknown model type.")

    # Create a pipeline with scaling and the model
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

    # 10-fold Cross Validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scoring = ['accuracy', 'roc_auc']

    cv_results = cross_validate(
        model_pipeline, X, y, cv=cv, scoring=scoring, return_estimator=True
    )

    avg_accuracy = np.mean(cv_results['test_accuracy'])
    avg_auc = np.mean(cv_results['test_roc_auc'])

    # Map feature names to importance scores
    feature_names = ["Size", "Depth", "Breadth", "Wiener Index", "Duration", "Acceleration"]
    importances = []

    for est in cv_results['estimator']:
        model = est.named_steps['classifier']
        if model_type in ['logistic', 'svm']:
            importances.append(model.coef_[0])
        elif model_type == 'rf':
            importances.append(model.feature_importances_)

    avg_importance = np.mean(importances, axis=0)
    feat_importance = dict(zip(feature_names, avg_importance))

    return {
        "accuracy": avg_accuracy,
        "auc": avg_auc,
        "feature_importance": feat_importance,
        "n_samples": len(y),
        "positive_ratio": float(np.mean(y))
    }
