import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv(r"./sets_split/train_set.csv", index_col = 0)
test_data = pd.read_csv(r"./sets_split/test_set.csv",index_col = 0)

measures = list(train_data.columns)
for column in ['crop_path','painting_name', 'directory','origin_painting','class', 'set', 'classification']:
    measures.remove(column)

train_data.loc[train_data['class'] == 'reproduction', 'class'] ='homografia'
train_data.loc[train_data['class'] == 'false', 'class'] ='brak powiązania'
train_data.loc[train_data['class'] == 'identity', 'class'] ='identyczność'
sns.set_theme(style = "whitegrid")
for measure in measures:
    dist_plot = sns.displot(train_data, x = measure, col = "class",facet_kws=dict(margin_titles=True),bins = 20, kde = True,
                       stat = "probability",common_bins = False, common_norm = False)
    plt.savefig(f'./plots/dist_plots/lin_scale/dist_plot_{measure}.png')
    plt.clf()
    try:
        dist_plot = sns.displot(train_data.dropna(), x=measure, col="class", facet_kws=dict(margin_titles=True), bins=20, kde=True,
                            stat="probability", common_bins=False, common_norm=False, log_scale = True,hue_order=['brak powiązania','homografia','identyczność'])
        plt.savefig(f'./plots/dist_plots/log_scale/dist_plot_{measure}.png')
    except:
        pass
    plt.clf()
    boxplot = sns.boxplot(x="class", y=measure,sym='',
                hue="class", palette=["r", "b", "g"], order = ['brak powiązania','homografia','identyczność'], hue_order=['brak powiązania','homografia','identyczność'],
                data=train_data)
    plt.legend(loc='upper right')
    plt.savefig(f'./plots/boxplots/fliers/boxplot_{measure}.png')
    plt.clf()
    boxplot = sns.boxplot(x="class", y=measure, sym='',
                          hue="class", palette=["r", "b", "g"], order=['brak powiązania','homografia','identyczność'],
                          hue_order=['brak powiązania','homografia','identyczność'],showfliers = False,
                          data=train_data)
    plt.legend(loc='upper right')
    plt.savefig(f'./plots/boxplots/no_fliers/boxplot_{measure}.png')