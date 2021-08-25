import pandas as pd
from numpy.random import choice
import numpy as np

results = pd.read_csv("test.csv")

results['directory'] = results['crop_path'].apply(lambda path: path.split("\\")[2])
results['origin_painting'] = results['crop_path'].apply(lambda path: path.split("\\")[3])

results['class'] = 'false'
results.loc[(results['directory'] == 'paintings') & (
        results['origin_painting'] == results['painting_name']), 'class'] = 'identity'
results.loc[(results['directory'] == 'reproductions') & (
        results['origin_painting'] == results['painting_name']), 'class'] = 'reproduction'

print(len(results[results['class'] == 'identity']))
print(len(results[results['class'] == 'reproduction']))
print(len(results[results['class'] == 'false']))

# rozlosować spośród false 4060 czy 4799?, z tego rozbicie na treningowy i testowy

results = pd.concat([results[results['class'] == 'false'].sample(
    n=len(results[results['class'] == 'reproduction']) + len(results[results['class'] == 'identity'])),
    results[results['class'] != 'false']])
'''
results_ver_2 = pd.concat([results[results['class'] == 'false'].sample(
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
