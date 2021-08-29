import pandas as pd
import numpy as np
from joblib import load
import os

validation_data = pd.read_csv(r"./sets_split/validation_set.csv", index_col=0)
sample_size = validation_data.shape[0]

positives_sample_size = np.sum(validation_data['classification'] == 1)
negatives_sample_size = np.sum(validation_data['classification'] == 0)

validation_data.dropna(inplace=True)
validation_data.reset_index(inplace=True)

y_val = validation_data['classification']


def false_positives(predictions, real_values):
    comparsion = pd.concat([predictions, real_values], axis=1)
    return np.sum((comparsion['prediction'] == 1) & (comparsion['classification'] == 0))


def false_negatives(predictions, real_values):
    comparsion = pd.concat([predictions, real_values], axis=1)
    return np.sum((comparsion['prediction'] == 0) & (comparsion['classification'] == 1))


model_names = os.listdir(r'./models/')
paths = [rf'./models/{model_name}' for model_name in iter(model_names)]

ttest_significant = ['no_of_matches', 'dist_median', 'dist_mean', 'dist_std', 'no_of_inliers',
                     'inliers_to_descriptors_ratio', 'matches_ratio']
model_variables = [['no_of_matches'], ['no_of_matches', 'homography_det']] + [ttest_significant for i in range(5)] + [
    ['dist_median', 'inliers_to_descriptors_ratio'] for i in range(5)]

errors = pd.DataFrame()

for model_ind, model_path in enumerate(paths):
    model = load(model_path)
    X_val = validation_data[model_variables[model_ind]].to_numpy()
    predictions = pd.Series(model.predict(X_val), name='prediction')
    FP = false_positives(predictions, y_val)
    FN = false_negatives(predictions, y_val)
    errors.loc[model_ind, 'model'] = model_path
    errors.loc[model_ind, 'false_positives'] = FP
    errors.loc[model_ind, 'false_negatives'] = FN
    errors.loc[model_ind, 'all'] = FP + FN

# no model
no_model_false_positives = np.sum(
    (validation_data['homography_det'].notna()) & (validation_data['classification'] == 0))
no_model_false_negatives = np.sum((validation_data['homography_det'].isna()) & (validation_data['classification'] == 1))

last_ind = errors.shape[0]
errors.loc[last_ind,'model']='no_model'
errors.loc[last_ind,'false_positives']=no_model_false_positives
errors.loc[last_ind,'false_negatives']=no_model_false_negatives
errors.loc[last_ind,'all']=no_model_false_negatives+no_model_false_positives

errors['FP_rate'] = errors['false_positives'] / negatives_sample_size
errors['FN_rate'] = errors['false_negatives'] / positives_sample_size
errors['accuracy'] = 1 - errors['all'] / sample_size

errors.to_csv('./errors/validation_errors.csv')
