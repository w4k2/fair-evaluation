from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats

np.set_printoptions(precision=3)

# Prepare classifiers
clfs = {
    "GNB": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "DTC": DecisionTreeClassifier(random_state=42),
}
clfns = ["gnb", "knn", "dtc"]
# Classifier combinations
clf_comb = list(combinations(range(len(clfs)), 2))

# Parameters
k_folds = 5
n_iters = 10
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

treshold = 2.7764

# p = (1 - stats.t.cdf(abs(2.7764), 4)) * 2
# print(p)
# exit()

for db_id, dataset in enumerate(datasets):
    # print("\n| %i/%i %s" % (db_id + 1, len(datasets), dataset))

    print(
        "\\begin{tabular}{c||cc||c|c|c||c}\n\t\\toprule\n\t\\multicolumn{7}{c}{\\textsc{%s}}\\\\"
        % dataset
    )

    # Load storage
    # tests = np.zeros((n_iters, len(clf_comb), len(alphas))).astype(int)
    # ps = np.zeros((n_iters, len(clf_comb)))
    # results = np.zeros((n_iters, len(clfs)))

    npzfile = np.load("results/%s.npz" % dataset)
    results = npzfile["results"]
    ps = npzfile["ps"]
    ts = npzfile["ts"]
    tests = npzfile["tests"]

    # Using only .05
    for p_id, pair in enumerate(clf_comb):
        print(
            "\t\\bottomrule\n\t\\multicolumn{7}{c}{\\includegraphics[width=7.5cm, trim=30 0 30 0]{figures/%s_%i.eps}}\\\\\n"
            % (dataset, p_id)
        )
        print(
            "\\midrule\t&\\textsc{%s} & \\textsc{%s} & \\textsc{t} & p & \\textsc{i} & \\textsc{d}\\\\"
            % (clfns[pair[0]], clfns[pair[1]])
        )
        # Get pure outcomes with distribution
        alpha_tests = tests[:, p_id, 1]
        pair_ts = ts[:, p_id]
        pair_ps = ps[:, p_id]
        outcomes, distribution = np.unique(alpha_tests, return_counts=True)
        distribution = distribution / np.sum(distribution)

        # Calculate T histogram
        # diff = (bins[1] - bins[0]) / 2
        # print(diff)
        # hist = np.histogram(pair_ts, bins=bins)[0]
        # print(hist)
        # print(bins)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_yticks([])
        # ax.plot(bins[:-1] + diff, hist, c="k")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(color="#E0E0E0", linestyle="-", linewidth=0.25)

        # Show mean result
        mean_t, mean_p = (np.median(pair_ts), np.median(pair_ps))
        mean_results = np.median(np.median(results[:, pair], axis=0), axis=1)
        std_t, std_p = np.std(pair_ts), np.std(pair_ps)
        color = "black"
        if mean_t > 0 and mean_p < 0.05:
            color = "red"
        elif mean_t < 0 and mean_p < 0.05:
            color = "blue"
        print(
            "\t\\color{%s} $\\approx$ & \\color{%s} %s %.3f &\\color{%s}  %s %.3f & %.2f & %.2f & --- & ---\\\\\\midrule"
            % (
                color,
                color,
                ("\\bfseries" if (mean_t > 0 or mean_p > 0.05) else ""),
                mean_results[0],
                color,
                ("\\bfseries" if (mean_t < 0 or mean_p > 0.05) else ""),
                mean_results[1],
                mean_t,
                mean_p,
            )
        )

        def point(x, text, color, marker="o"):
            ax.plot(
                [x if np.abs(x) < 10 else (10 if x > 0 else -10)],
                [0.45],
                marker,
                c=color,
            )
            ax.annotate(
                "",
                xy=(x if np.abs(x) < 10 else (10 if x > 0 else -10), 0.39),
                horizontalalignment="center",
                fontsize=8,
                c=color,
            )

        ax.axvline(
            x=mean_t if np.abs(mean_t) < 10 else (10 if mean_t > 0 else -10), c=color
        )

        txt = "T=%3.2f [+-%.2f]\np=%3.2f [+-%.2f]" % (mean_t, std_t, mean_p, std_p)
        point(mean_t, "                m|%.2f" % mean_t, color, marker="D")

        # Density estimation
        est = KernelDensity(bandwidth=1).fit(pair_ts.reshape(-1, 1))
        hmax, hbin = (10, 500)
        # ax.set_xticks(np.linspace(-hmax, hmax, 9))
        ax.set_xticks([-hmax, -treshold, 0, treshold, hmax])
        ax.set_xlim(-hmax, hmax)

        # Line
        bins = np.linspace(-hmax, hmax, hbin)
        log_dens = est.score_samples(bins.reshape(-1, 1))
        nexp_dens = np.exp(log_dens)
        nexp_dens[0], nexp_dens[-1] = (0, 0)
        ax.plot(bins, nexp_dens, c=color)

        # Left tail
        bins = np.linspace(-hmax, -treshold, hbin)
        log_dens = est.score_samples(bins.reshape(-1, 1))
        nexp_dens = np.exp(log_dens)
        nexp_dens[0], nexp_dens[-1] = (0, 0)
        ax.fill(bins, nexp_dens, c="b")
        ax.set_ylim(0, 0.5)

        # Dependent
        bins = np.linspace(-treshold, treshold, hbin)
        # print([bins] + treshold)
        # exit()
        log_dens = est.score_samples(bins.reshape(-1, 1))
        nexp_dens = np.exp(log_dens)
        nexp_dens[0] = 0
        nexp_dens[-1] = 0
        # ax.fill(bins, nexp_dens, c="k")

        # Right tail
        bins = np.linspace(treshold, hmax, hbin)
        log_dens = est.score_samples(bins.reshape(-1, 1))
        nexp_dens = np.exp(log_dens)
        nexp_dens[0], nexp_dens[-1] = (0, 0)
        ax.fill(bins, nexp_dens, c="r")
        # ax.set_title("T-statistic distribution", fontsize=8)

        def show_instance(i, label="0", bold=[False, False]):
            mean_results = np.mean(results[i], axis=1)
            d = 0
            color = "black"
            if label == "-":
                d = distribution[np.where(outcomes == 2)[0]]
                color = "blue"
            elif label == "=":
                d = distribution[np.where(outcomes == 0)[0]]
            elif label == "+":
                d = distribution[np.where(outcomes == 1)[0]]
                color = "red"

            point(pair_ts[i], "%.2f\n" % pair_ts[i], color)

            # print(d)
            print(
                "\t{\\bfseries\\color{%s}\\tiny%s}& \\color{%s} %s %.3f & \\color{%s} %s %.3f & %.2f & %.2f & %i & \\color{%s} %.4f\\\\"
                % (
                    color,
                    label,
                    color,
                    "\\bfseries" if bold[0] else "",
                    mean_results[pair[0]],
                    color,
                    "\\bfseries" if bold[1] else "",
                    mean_results[pair[1]],
                    pair_ts[i],
                    pair_ps[i],
                    i,
                    color,
                    d,
                )
            )

        def show_no_instance(label="0"):
            print("\t{\\tiny%s}& --- & --- & --- & --- & --- & ---\\\\" % (label))

        if 0 in outcomes:
            # Lowest absolute T
            i = np.argmin(np.abs(pair_ts))
            show_instance(i, "=", [True, True])
        else:
            show_no_instance("=")
        if 2 in outcomes:
            # Lowest T
            i = np.argmin(pair_ts)
            show_instance(i, "-", [False, True])
        else:
            show_no_instance("-")
        if 1 in outcomes:
            # Highest T
            i = np.argmax(pair_ts)
            show_instance(i, "+", [True, False])
        else:
            show_no_instance("+")

        plt.tight_layout()
        plt.savefig("foo.png")
        plt.savefig("figures/%s_%i.png" % (dataset, p_id))
        plt.savefig("figures/%s_%i.eps" % (dataset, p_id))
        plt.clf()
        plt.close("all")

    print("\\bottomrule\\end{tabular}\n\n")
    # exit()
