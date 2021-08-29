import pandas as pd
from numpy.random import choice
import numpy as np

results = pd.read_csv("test.csv", index_col=0)

results['directory'] = results['crop_path'].apply(lambda path: path.split("\\")[2])
results['origin_painting'] = results['crop_path'].apply(lambda path: path.split("\\")[3])

results['class'] = 'false'
results.loc[(results['directory'] == 'paintings') & (
        results['origin_painting'] == results['painting_name']), 'class'] = 'identity'
results.loc[(results['directory'] == 'reproductions') & (
        results['origin_painting'] == results['painting_name']), 'class'] = 'reproduction'

results['inliers_ratio'] = results['no_of_inliers'] / (results['no_of_outliers'] + results['no_of_inliers'])
results['matches_ratio'] = results['no_of_matches'] / results['no_of_descriptors']
results['inliers_to_descriptors_ratio'] = results['no_of_inliers'] / results['no_of_descriptors']

ratio_columns = ['no_of_matches', 'no_of_inliers', 'matches_ratio', 'inliers_ratio', 'inliers_to_descriptors_ratio']
results[ratio_columns] = results[ratio_columns].fillna(value=0)

results['classification'] = 1
results.loc[results['class'] == 'false', 'classification'] = 0

'''
results = pd.concat([results[results['class'] == 'false'].sample(
    n=len(results[results['class'] == 'identity'])), results[results['class'] == 'reproduction'].sample(
    n=len(results[results['class'] == 'identity'])),
    results[results['class'] == 'identity']])
'''
results = pd.concat([results[results['class'] == 'false'].sample(
    n=len(results[results['class'] == 'reproduction']) + len(results[results['class'] == 'identity'])),
    results[results['class'] == 'reproduction'],
    results[results['class'] == 'identity']])

# zbiór treningowy i testowy

results['set'] = np.nan
results.loc[results['class'] == 'identity', 'set'] = choice(a=['train', 'validation', 'test'], size=len(
    results.loc[results['class'] == 'identity', 'set']), p=[0.64, 0.16, 0.2])
results.loc[results['class'] == 'reproduction', 'set'] = choice(a=['train', 'validation', 'test'], size=len(
    results.loc[results['class'] == 'reproduction', 'set']), p=[0.64, 0.16, 0.2])
results.loc[results['class'] == 'false', 'set'] = choice(a=['train', 'validation', 'test'],
                                                         size=len(results.loc[results['class'] == 'false', 'set']),
                                                         p=[0.64, 0.16, 0.2])

train_set = results[results['set'] == 'train']
validation_set = results[results['set'] == 'validation']
test_set = results[results['set'] == 'test']

train_set.to_csv("train_set.csv")
validation_set.to_csv("validation_set.csv")
test_set.to_csv("test_set.csv")
