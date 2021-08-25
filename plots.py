import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv(r"train_set.csv", index_col = 0)
test_data = pd.read_csv(r"test_set.csv",index_col = 0)

measures = list(train_data.columns)
for column in ['crop_path','painting_name', 'directory','origin_painting','class', 'set']:
    measures.remove(column)

sns.set_theme(style = "whitegrid")
for measure in measures:
    dist_plot = sns.displot(train_data, x = measure, col = "class",facet_kws=dict(margin_titles=True),bins = 20, kde = True,
                       stat = "probability",common_bins = False, common_norm = False)
    plt.savefig(f'./plots/dist_plots/lin_scale/dist_plot_{measure}.png')
    plt.clf()
    try:
        dist_plot = sns.displot(train_data, x=measure, col="class", facet_kws=dict(margin_titles=True), bins=20, kde=True,
                            stat="probability", common_bins=False, common_norm=False, log_scale = True)
        plt.savefig(f'./plots/dist_plots/log_scale/dist_plot_{measure}.png')
    except:
        pass
    plt.clf()
    boxplot = sns.boxplot(x="class", y=measure,sym='',
                hue="class", palette=["r", "b", "g"], order = ['false','reproduction','identity'], hue_order=['false','reproduction','identity'],
                data=train_data)
    plt.savefig(f'./plots/boxplots/fliers/boxplot_{measure}.png')
    plt.clf()
    boxplot = sns.boxplot(x="class", y=measure, sym='',
                          hue="class", palette=["r", "b", "g"], order=['false', 'reproduction', 'identity'],
                          hue_order=['false', 'reproduction', 'identity'],showfliers = False,
                          data=train_data)
    plt.savefig(f'./plots/boxplots/no_fliers/boxplot_{measure}.png')