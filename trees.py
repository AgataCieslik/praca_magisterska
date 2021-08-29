from sklearn import tree
import pandas as pd
from joblib import dump

data = pd.read_csv(r"train_set.csv", index_col=0)

clf_depth_1 = tree.DecisionTreeClassifier(max_depth=1)
clf_depth_2 = tree.DecisionTreeClassifier(max_depth=2)
clf_depth_3 = tree.DecisionTreeClassifier(max_depth=3)
clf_depth_5 = tree.DecisionTreeClassifier(max_depth=5)
clf_depth_10 = tree.DecisionTreeClassifier(max_depth=10)

# model 0: no_of_matches
subset = data[['no_of_matches', 'classification']].dropna()
X = subset['no_of_matches'].to_numpy().reshape(-1, 1)
y = subset['classification'].to_numpy()
model_0 = clf_depth_1.fit(X, y)
dump(model_0, '.\models\model_0.joblib')

# model 1: homography_det + no_of_matches
subset = data[['no_of_matches','homography_det', 'classification']].dropna()
X = subset[['no_of_matches','homography_det']].to_numpy()
y = subset['classification'].to_numpy()
model_1 = clf_depth_2.fit(X,y)
dump(model_1, '.\models\model_1.joblib')

# model 2: all t-test significant
ttest_significant = ['no_of_matches','dist_median','dist_mean','dist_std','no_of_inliers', 'inliers_to_descriptors_ratio','matches_ratio']
subset = data[ttest_significant+['classification']].dropna()
X = subset[ttest_significant].to_numpy()
y = subset['classification'].to_numpy()

model_2_1 = clf_depth_1.fit(X,y)
model_2_2 = clf_depth_2.fit(X,y)
model_2_3 = clf_depth_3.fit(X,y)
model_2_5 = clf_depth_5.fit(X,y)
model_2_10 = clf_depth_10.fit(X,y)

dump(model_2_1,'.\models\model_2_1.joblib')
dump(model_2_2,'.\models\model_2_2.joblib')
dump(model_2_3,'.\models\model_2_3.joblib')
dump(model_2_5,'.\models\model_2_5.joblib')
dump(model_2_10,'.\models\model_2_10.joblib')

# model 3: dist_median + inliers_to_descriptors_ratio
subset = data[['dist_median', 'inliers_to_descriptors_ratio','classification']].dropna()
X = subset[['dist_median', 'inliers_to_descriptors_ratio']].to_numpy()
y = subset['classification'].to_numpy()

model_3_1 = clf_depth_1.fit(X,y)
model_3_2 = clf_depth_2.fit(X,y)
model_3_3 = clf_depth_3.fit(X,y)
model_3_5 = clf_depth_5.fit(X,y)
model_3_10 = clf_depth_10.fit(X,y)

dump(model_3_1,'.\models\model_3_1.joblib')
dump(model_3_2,'.\models\model_3_2.joblib')
dump(model_3_3,'.\models\model_3_3.joblib')
dump(model_3_5,'.\models\model_3_5.joblib')
dump(model_3_10,'.\models\model_3_10.joblib')


