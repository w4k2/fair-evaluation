import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
from itertools import combinations
from tqdm import tqdm


def t_dep(a, b):
    # calculate means
    mean1, mean2 = np.mean(a), np.mean(b)

    # number of paired samples
    n = len(a)

    # sum squared difference between observations
    d1 = np.sum([(a[i] - b[i]) ** 2 for i in range(n)])

    # sum difference between observations
    d2 = np.sum([a[i] - b[i] for i in range(n)])

    # standard deviation of the difference between means
    sd = np.sqrt((d1 - (d2 ** 2 / n)) / (n - 1))

    # standard error of the difference between the means
    sed = sd / np.sqrt(n)

    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed

    if np.isnan(t_stat):
        return 0, 1

    # degrees of freedom
    df = n - 1

    # calculate the p-value
    p = (1 - stats.t.cdf(abs(t_stat), df)) * 2

    return t_stat, p


def load_dataset(dbname):
    df = pd.read_csv("datasets/%s.csv" % dbname)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


# Prepare classifiers
clfs = {
    "GNB": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "DTC": DecisionTreeClassifier(random_state=42),
}
# Classifier combinations
clf_comb = list(combinations(range(len(clfs)), 2))

# Parameters
k_folds = 5
n_iters = 10000
alphas = [0.1, 0.05, 0.01]
datasets = [
    "wisconsin",
    "wine",
    "soybean",
    "sonar",
    "monkthree",
    "monkone",
    "liver",
    "ionosphere",
    "heart",
    "hayes",
    "german",
    "cryotherapy",
    "breastcan",
    "banknote",
    "balance",
    "australian",
    "iris",
    "diabetes",
]

# Iterate datasets
for db_id, dataset in enumerate(datasets):
    print("%i/%i %s" % (db_id + 1, len(datasets), dataset))
    # Load dataset
    X, y = load_dataset(dataset)

    # Prepare storage
    tests = np.zeros((n_iters, len(clf_comb), len(alphas))).astype(int)
    ps = np.zeros((n_iters, len(clf_comb)))
    ts = np.zeros((n_iters, len(clf_comb)))
    results = np.zeros((n_iters, len(clfs), k_folds))
    mean_results = np.zeros((n_iters, len(clfs)))

    # Perform iterations
    for i in tqdm(range(n_iters), ascii=True):
        # Perform experiment on k-fold
        scores = np.zeros((len(clfs), k_folds))
        skf = StratifiedKFold(n_splits=k_folds, random_state=i, shuffle=True)
        for fold, (train, test) in enumerate(skf.split(X, y)):
            for clf_idx, clf_n in enumerate(clfs):
                clf = clone(clfs[clf_n])
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                score = accuracy_score(y_pred, y[test])
                scores[clf_idx, fold] = score

        # Get mean scores
        mean_scores = np.mean(scores, axis=1)
        mean_results[i] = mean_scores
        results[i] = scores

        # Iterate combinations
        for p_id, pair in enumerate(clf_comb):
            a = scores[pair[0]]
            b = scores[pair[1]]
            t, p = t_dep(a, b)

            ts[i, p_id] = t
            ps[i, p_id] = p

            # Analyze results
            # 0 bez różnic, 1 lepszy pierwszy, 2, drugi lepszy
            for a_id, alpha in enumerate(alphas):
                result = 0 if p > alpha else (1 if t > 0 else 2)
                # print("%.2f | %i | %.3f | %.3f vs %.3f" % (alpha, result, p, mean_scores[pair[0]], mean_scores[pair[1]]))
                tests[i, p_id, a_id] = result

    np.savez("results/%s" % dataset, results=results, tests=tests, ps=ps, ts=ts)
    # print(results, tests, ps)
