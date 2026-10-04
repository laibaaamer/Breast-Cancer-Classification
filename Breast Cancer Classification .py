import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay,precision_score,recall_score,f1_score
data = load_breast_cancer()
X = data.data
y = data.target
print("Dataset Shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])
param_grid = {
    "classifier__n_estimators":[100,200],
    "classifier__max_depth":[None,10],
    "classifier__min_samples_split":[2,5],
    "classifier__min_samples_leaf":[1,2]
}
param_dist = {
    "classifier__n_estimators": randint(100, 301),
    "classifier__max_depth": randint(5, 21),
    "classifier__min_samples_split": randint(2, 11),
    "classifier__min_samples_leaf": randint(1, 5)
}
start=time.time()
grid=GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train,y_train)
grid_time=time.time()-start
best_grid=grid.best_estimator_
grid_pred=best_grid.predict(X_test)
grid_acc=accuracy_score(y_test,grid_pred)
grid_pre=precision_score(y_test,grid_pred)
grid_rec=recall_score(y_test,grid_pred)
grid_f1=f1_score(y_test,grid_pred)
print("GridSearchCV")
print("Best Parameters:",grid.best_params_)
print("Best CV Score:",round(grid.best_score_,4))
print("Test Accuracy:",round(grid_acc,4))
print(classification_report(y_test,grid_pred))
start=time.time()
random=RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)
random.fit(X_train,y_train)
random_time=time.time()-start
best_random=random.best_estimator_
random_pred=best_random.predict(X_test)
random_acc=accuracy_score(y_test,random_pred)
random_pre=precision_score(y_test,random_pred)
random_rec=recall_score(y_test,random_pred)
random_f1=f1_score(y_test,random_pred)
print("RandomizedSearchCV")
print("Best Parameters:",random.best_params_)
print("Best CV Score:",round(random.best_score_,4))
print("Test Accuracy:",round(random_acc,4))
print(classification_report(y_test,random_pred))
ConfusionMatrixDisplay.from_predictions(y_test,grid_pred)
plt.title("GridSearchCV Confusion Matrix")
plt.show()
ConfusionMatrixDisplay.from_predictions(y_test,random_pred)
plt.title("RandomizedSearchCV Confusion Matrix")
plt.show()
importance=pd.Series(
    best_grid.named_steps["classifier"].feature_importances_,
    index=data.feature_names
).sort_values(ascending=False)
plt.figure(figsize=(10,5))
importance.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importance")
plt.tight_layout()
plt.show()
comparison=pd.DataFrame({
    "Metric":[
        "Best CV Score",
        "Test Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Training Time (sec)"
    ],
    "GridSearchCV":[
        round(grid.best_score_,4),
        round(grid_acc,4),
        round(grid_pre,4),
        round(grid_rec,4),
        round(grid_f1,4),
        round(grid_time,2)
    ],
    "RandomizedSearchCV":[
        round(random.best_score_,4),
        round(random_acc,4),
        round(random_pre,4),
        round(random_rec,4),
        round(random_f1,4),
        round(random_time,2)
    ]
})
print("Comparison")
print(comparison)