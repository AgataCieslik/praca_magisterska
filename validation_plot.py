import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from joblib import load

h = .02

names = ["Decision Tree depth=1", "Decision Tree depth=2","Decision Tree depth=3", "Decision Tree depth=5", "Decision Tree depth=10"]

classifiers = [
    load('.\models\model_3_1.joblib'),
    load('.\models\model_3_2.joblib'),
    load('.\models\model_3_3.joblib'),
    load('.\models\model_3_5.joblib'),
    load('.\models\model_3_10.joblib'),
]

validation_data_full = pd.read_csv("validation_set.csv", index_col=0)
sample_size = validation_data_full.shape[0]

validation_data = validation_data_full[validation_data_full['dist_median'].notna() & validation_data_full['inliers_to_descriptors_ratio'].notna()]

X_val = validation_data.loc[:, ['dist_median', 'inliers_to_descriptors_ratio']].to_numpy()

y_val = validation_data.loc[:, 'classification'].to_numpy()


figure = plt.figure(figsize=(24, 9))
i = 1

x_min, x_max = X_val[:, 0].min() - .5, X_val[:, 0].max() + .5
y_min, y_max = X_val[:, 1].min() - .5, X_val[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, len(classifiers) , i)


ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright, alpha=0.6,
           edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    mistakes = np.sum(clf.predict(X_val) != y_val)
    score = 1-mistakes/sample_size

    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright,
               edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlabel("distances median")
    ax.set_ylabel("inliers/descriptors")
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()
plt.savefig(f'./plots/validation_plot.png')