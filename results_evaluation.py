import pandas as pd
from os.path import join, isfile
from os import listdir, makedirs
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from config import get_result_path, DIR_PATH


def calculate_metrics_by_prefixes(prefix_results, path, set):
    # metriche generali dell'epoca per il set corrente
    accuracy = float(accuracy_score(prefix_results['y_real'].tolist(), prefix_results['y_estimated'].tolist()))
    f1_w = float(f1_score(prefix_results['y_real'].tolist(), prefix_results['y_estimated'].tolist(), average='weighted'))  # , zero_division=1))

    epoch_metrics = {
        'acc': round(accuracy * 100, 2),
        'f1': round(f1_w * 100, 2),
    }

    # ----
    # metriche dell'epoca al variare della lunghezza del prefisso per il set corrente
    columns = ['set', 'prefix_len', 'acc', 'f1']
    df_set = pd.DataFrame(columns=columns)

    prefix_lens = sorted(prefix_results['prefix_len'].unique().tolist())
    for size in prefix_lens:
        size_df = prefix_results.loc[prefix_results['prefix_len'] == size]

        accuracy = float(accuracy_score(size_df['y_real'].tolist(), size_df['y_estimated'].tolist()))
        f1_w = float(f1_score(size_df['y_real'].tolist(), size_df['y_estimated'].tolist(),
                              average='weighted'))  # , zero_division=1)

        df_set.loc[len(df_set)] = [f'{set}', size, round(accuracy * 100, 2), round(f1_w * 100, 2)]

    df_set.to_csv(join(path, f'prefix_metrics_{set}.csv'), header=True, sep=',', index=False)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    ax1.set_title(f"Accuracy varying prefix size on {set} set")
    ax1.set_xlabel(f"Prefix size")
    ax1.set_ylabel(f"Accuracy")
    ax1.set_xticks(prefix_lens)
    ax1.plot(df_set['prefix_len'], df_set['acc'])

    ax2.set_title(f"Weighted F1 varying prefix size on {set} set")
    ax2.set_xlabel(f"Prefix size")
    ax2.set_ylabel(f"Weighted F1")
    ax2.set_xticks(prefix_lens)
    ax2.plot(df_set['prefix_len'], df_set['f1'])

    plt.tight_layout()
    plt.savefig(join(path, f'prefix_metrics_{set}.png'))
    plt.close('all')
    # ---
    return epoch_metrics


def calculate_combination_metrics(results, path):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 6))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    ax1.set_title(f"Accuracy on train set")
    ax1.set_xlabel(f"Epochs")
    ax1.set_ylabel(f"Accuracy")
    ax1.plot(results['epoch'], results['train_acc'])

    ax4.set_title(f"Accuracy on test set")
    ax4.set_xlabel(f"Epochs")
    ax4.set_ylabel(f"Accuracy")
    ax4.plot(results['epoch'], results['test_acc'])

    ax2.set_title(f"Weighted F1 on train set")
    ax2.set_xlabel(f"Epochs")
    ax2.set_ylabel(f"Weighted F1")
    ax2.plot(results['epoch'], results['train_f1'])

    ax5.set_title(f"Weighted F1 on test set")
    ax5.set_xlabel(f"Epochs")
    ax5.set_ylabel(f"Weighted F1")
    ax5.plot(results['epoch'], results['test_f1'])

    ax3.set_title(f"Loss on train set")
    ax3.set_xlabel(f"Epochs")
    ax3.set_ylabel(f"Loss")
    ax3.plot(results['epoch'], results['train_loss'])

    ax6.set_title(f"Loss on test set")
    ax6.set_xlabel(f"Epochs")
    ax6.set_ylabel(f"Loss")
    ax6.plot(results['epoch'], results['test_loss'])

    plt.tight_layout()
    plt.savefig(join(path, f'combination_metrics.png'))
    plt.close('all')


def best_metric_on_set(df, ds_name, model, metric='loss', set='train'):
    path_to_save = join(DIR_PATH, 'results', 'best', model)
    makedirs(path_to_save, exist_ok=True)
    best_df = DataFrame()
    on_metric = f'{set}_{metric}'
    for comb in df['combination'].unique().tolist():
        comb_df = df.loc[df['combination'] == comb]
        if metric == 'loss':
            avg_best = comb_df.loc[(comb_df[on_metric] == comb_df[on_metric].min())]
        else:
            avg_best = comb_df.loc[(comb_df[on_metric] == comb_df[on_metric].max())]

        avg_best = avg_best.copy()
        avg_best['type_best'] = f'best_{on_metric}'
        #avg_best.loc[:, 'type_best'] = f'best_{on_metric}'
        best_df = pd.concat([best_df, avg_best])

    ascending = True if metric == 'loss' else False
    best_df.sort_values(by=[on_metric], ascending=ascending, inplace=True)
    best_df.to_csv(join(path_to_save, f'{ds_name}_best_{on_metric}.csv'), header=True, index=False, sep=',')


def plot_confusion_matrix(prefix_results, ds_activities, epoch_path, set):
    plt.figure(figsize=(15, 6))
    activities = []
    with open(ds_activities, 'r') as f:
        for lines in f.readlines():
            lines = lines[:-1]
            activities.append(lines)

    activity_names = {idx: activity for idx, activity in enumerate(sorted(activities))}

    true_labels = prefix_results["y_real"].replace(activity_names)
    predicted_labels = prefix_results["y_estimated"].replace(activity_names)
    cm = confusion_matrix(true_labels.astype(str), predicted_labels.astype(str))
    cmd = ConfusionMatrixDisplay(cm)

    cmd.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(join(epoch_path, f'confusion_matrix_{set}.png'), pad_inches=5)
    plt.close('all')



def best_combinations(model, ds_name):
    results_path = get_result_path(ds_name, model)
    df_results = pd.DataFrame()
    runs = list(listdir(results_path))
    for run in runs:
        # results/HD/YYYY-MM-DD_HH_mm_ss
        if run == '.DS_Store':
            continue
        combinations = list(listdir(join(results_path, run)))
        for combination in combinations:
            # results/HD/YYYY-MM-DD_HH_mm_ss/comb
            if combination == '.DS_Store':
                continue
            comb_path = join(results_path, run, combination)
            comb_items = list(listdir(comb_path))
            # results/HD/YYYY-MM-DD_HH_mm_ss/comb/epoch
            for comb_item in comb_items:
                if comb_item == '.DS_Store':
                    continue
                if isfile(join(comb_path, comb_item)) and comb_item.startswith('results'):
                    # results/HD/YYYY-MM-DD_HH_mm_ss/comb/results.csv
                    comb_results = pd.read_csv(join(comb_path, comb_item), header=0)
                    df_results = pd.concat([df_results, comb_results])

    best_metric_on_set(df_results, ds_name, model)
    best_metric_on_set(df_results, ds_name, model, metric='acc')
    best_metric_on_set(df_results, ds_name, model, metric='f1')
