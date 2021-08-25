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


results['inliers_ratio'] = results['no_of_inliers']/(results['no_of_outliers']+results['no_of_inliers'])
results['matches_ratio'] = results['no_of_matches']/results['no_of_descriptors']
results['inliers_to_descriptors_ratio'] = results['no_of_inliers']/results['no_of_descriptors']

ratio_columns = ['no_of_matches','no_of_inliers','matches_ratio','inliers_ratio','inliers_to_descriptors_ratio']
results[ratio_columns] = results[ratio_columns].fillna(value=0)

'''
results = pd.concat([results[results['class'] == 'false'].sample(
    n=len(results[results['class'] == 'identity'])), results[results['class'] == 'reproduction'].sample(
    n=len(results[results['class'] == 'identity'])),
    results[results['class'] == 'identity']])
'''

# zbiór treningowy i testowy
results['set'] = np.nan
results.loc[results['class'] == 'identity', 'set'] = choice(a=['train', 'test'], size=len(
    results.loc[results['class'] == 'identity', 'set']), p=[3 / 4, 1 / 4])
results.loc[results['class'] == 'reproduction', 'set'] = choice(a=['train', 'test'], size=len(
    results.loc[results['class'] == 'reproduction', 'set']), p=[3 / 4, 1 / 4])
results.loc[results['class'] == 'false', 'set'] = choice(a=['train', 'test'],
                                                         size=len(results.loc[results['class'] == 'false', 'set']),
                                                         p=[3 / 4, 1 / 4])

train_set = results[results['set'] == 'train']
test_set = results[results['set'] == 'test']

train_set.to_csv("train_set.csv")
test_set.to_csv("test_set.csv")
