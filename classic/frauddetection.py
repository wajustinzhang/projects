import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)
n_samples = 200
outlier_fraction=0.25
clusters_seperation=[0,1,2]

classfiers={
    "one-class SVM": svm.OneClassSVM(nu=0.95*outlier_fraction + 0.05, kernel="rbf", gamma=0.01),
    "Robust covariance": EllipticEnvelope(contamination=outlier_fraction),
    "Isolation Forest": IsolationForest(max_samples=n_samples, contamination=outlier_fraction, random_state=rng)
}

xx, yy = np.meshgrid(np.linspace(-7,7,500), np.linspace(-7,7,500))
n_inliers = int((1.-outlier_fraction)*n_samples)
n_outliers = int((outlier_fraction)*n_samples)
ground_truth=np.ones(n_samples, dtype=int)
ground_truth[-n_outliers:]=-1

for i, offset in enumerate(clusters_seperation):
    np.random.seed(42)
    X1=0.3*np.random.randn(n_inliers//2, 2) - offset
    X2=0.3*np.random.randn(n_inliers//2, 2) + offset
    X=np.r_[X1,X2]
    plt.plot(X1, X2, 'ro')
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

    # Fit the model
    plt.figure(figsize=(10.8, 3.6))
    for i, (clf_name, clf) in enumerate(classfiers.items()):
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        threshold = stats.scoreatpercentile(scores_pred,
                                            100 * outlier_fraction)
        y_pred = clf.predict(X)
        n_errors = (y_pred != ground_truth).sum()
        # plot the levels lines and the points
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        subplot = plt.subplot(1, 3, i + 1)
        subplot.contourf(xx, yy,Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)
        a=subplot.contour(xx, yy, Z, levels=[threshold], lineswidths=2, colors='red')
        subplot.contourf(xx,yy,Z,levels=[threshold, Z.max()], colors='orange')
        b=subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white')
        c=subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='black')
        subplot.axis('tight')
        subplot.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'true inliers', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=11),
            loc='lower right')
        subplot.set_title("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        subplot.set_xlim((-7, 7))
        subplot.set_ylim((-7, 7))

    plt.subplots_adjust(0.04, 0.1, 0.96, 0.92, 0.1, 0.26)

plt.show()