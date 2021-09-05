import os
from sklearn import tree
from joblib import load
import matplotlib.pyplot as plt

model_names = os.listdir(r'./models/')
paths = [rf'./models/{model_name}' for model_name in iter(model_names)]

# model 0: no_of_matches
subset = ['no_of_matches']
model_0 = load('.\models\model_0.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_0, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_0.jpg")

# model 1: homography_det + no_of_matches
subset = ['no_of_matches', 'homography_det']
model_1 = load('.\models\model_1.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_1, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_1.jpg")

# model 2: all t-test significant
subset = ['no_of_matches', 'dist_median', 'dist_mean', 'dist_std', 'no_of_inliers', 'inliers_to_descriptors_ratio',
          'matches_ratio']

model_2_1 = load('.\models\model_2_1.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_2_1, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_2_1.jpg")

model_2_2 = load('.\models\model_2_2.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_2_2, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_2_2.jpg")

model_2_3 = load('.\models\model_2_3.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_2_3, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_2_3.jpg")

model_2_5 = load('.\models\model_2_5.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_2_5, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_2_5.jpg")

model_2_10 = load('.\models\model_2_10.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_2_10, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_2_10.jpg")

# model 3: dist_median + inliers_to_descriptors_ratio
subset = ['dist_median', 'inliers_to_descriptors_ratio']

model_3_1 = load('.\models\model_3_1.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_3_1, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_3_1.jpg")

model_3_2 = load('.\models\model_3_2.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_3_2, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_3_2.jpg")

model_3_3 = load('.\models\model_3_3.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_3_3, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_3_3.jpg")

model_3_5 = load('.\models\model_3_5.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_3_5, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_3_5.jpg")

model_3_10 = load('.\models\model_3_10.joblib')
plt.figure(figsize=[20, 20])
tree.plot_tree(model_3_10, filled=True, feature_names=subset)
plt.savefig(r"./plots/tree_schemas/tree_schema_model_3_10.jpg")
