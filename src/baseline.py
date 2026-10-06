"""Step 2: baselines with TF-IDF features (Logistic Regression and Naive Bayes)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from utils import load_splits, compute_metrics_dict, log_experiment, save_predictions

train, val, test, labels = load_splits()

vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
Xtr = vec.fit_transform(train.text)
Xva, Xte = vec.transform(val.text), vec.transform(test.text)

models = {
    "tfidf_logreg": (LogisticRegression(max_iter=2000, C=10), "TF-IDF 1-2gram 50k, LogReg C=10"),
    "tfidf_nb": (MultinomialNB(alpha=0.1), "TF-IDF 1-2gram 50k, MultinomialNB alpha=0.1"),
}

for name, (clf, config) in models.items():
    clf.fit(Xtr, train.label)
    print(f"\n=== {name} ===")
    print("Validation:", compute_metrics_dict(val.label, clf.predict(Xva)))
    pred = clf.predict(Xte)
    print(classification_report(test.label, pred, target_names=labels, zero_division=0))
    log_experiment(name, config, compute_metrics_dict(test.label, pred))
    save_predictions(name, test.text, test.label, pred, labels)
