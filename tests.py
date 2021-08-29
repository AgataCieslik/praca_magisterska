import pandas as pd
import scipy.stats as stats

data = pd.read_csv(r"train_set.csv", index_col=0)
numeric_columns = ['no_of_matches', 'dist_median', 'dist_mean', 'dist_std', 'no_of_inliers', 'homography_norm',
                   'homography_det', 'inliers_to_descriptors_ratio', 'matches_ratio']

test_statistics = pd.DataFrame()
test_statistics['metric'] = numeric_columns

# normality test (D'Agostino, Pearson)
test_statistics['normaltest_group_false'] = [stats.normaltest(data[col][data['class'] == 'false']).pvalue for col in
                                             iter(numeric_columns)]
test_statistics['normaltest_group_identity'] = [stats.normaltest(data[col][data['class'] == 'identity']).pvalue for col
                                                in
                                                iter(numeric_columns)]
test_statistics['normaltest_group_reproduction'] = [stats.normaltest(data[col][data['class'] == 'reproduction']).pvalue
                                                    for
                                                    col in iter(numeric_columns)]

# Kruskal-Wallis test
kruskal_results = [stats.kruskal(data[col][data['class'] == 'false'].dropna(),
                                 data[col][data['class'] == 'reproduction'].dropna(),
                                 data[col][data['class'] == 'identity'].dropna()) for col in iter(numeric_columns)]

test_statistics['kruskal_statistic'] = [result.statistic for result in iter(kruskal_results)]
test_statistics['kruskal_pvalue'] = [result.pvalue for result in iter(kruskal_results)]

# T test

# two-sided
two_sided_results = [stats.ttest_ind(data[col][data['class'] == 'false'].dropna(),
                                     data[col][data['class'] == 'reproduction'].dropna(), equal_var=False,
                                     alternative="two-sided") for col in iter(numeric_columns)]

test_statistics['two-sided_statistic'] = [result.statistic for result in iter(two_sided_results)]
test_statistics['two-sided_pvalue'] = [result.pvalue for result in iter(two_sided_results)]

# greater
greater_results = [stats.ttest_ind(data[col][data['class'] == 'false'].dropna(),
                                   data[col][data['class'] == 'reproduction'].dropna(), equal_var=False,
                                   alternative="greater") for col in iter(numeric_columns)]

test_statistics['greater_statistic'] = [result.statistic for result in iter(greater_results)]
test_statistics['greater_pvalue'] = [result.pvalue for result in iter(greater_results)]

# less
less_results = [stats.ttest_ind(data[col][data['class'] == 'false'].dropna(),
                                data[col][data['class'] == 'reproduction'].dropna(), equal_var=False,
                                alternative="less") for col in iter(numeric_columns)]

test_statistics['less_statistic'] = [result.statistic for result in iter(less_results)]
test_statistics['less_pvalue'] = [result.pvalue for result in iter(less_results)]

test_statistics.to_csv(r"./results/tests_results.csv")

means = data[numeric_columns + ['class']].groupby('class').mean().add_suffix("_mean")
variances = data[numeric_columns + ['class']].groupby('class').var().add_suffix("_variance")
skewness = data[numeric_columns + ['class']].groupby('class').agg(stats.skew).add_suffix("_skewness")
kurtosis = data[numeric_columns + ['class']].groupby('class').agg(stats.kurtosis).add_suffix("_kurtosis")
statistics = pd.concat([means, variances, skewness, kurtosis], axis=1)
statistics.to_csv(r"./results/stats.csv")
