"""Few-shot ablation study"""
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
from visualize_utils import *


################################################################################################
#### Plotting helper                                                                        ####
################################################################################################

rcParams['figure.dpi'] = 500
rcParams['savefig.dpi'] = 500
rcParams['figure.figsize'] = 9, 24
rcParams['font.family'] = 'Arial'
rcParams['axes.labelsize'] = 18
rcParams['axes.titlesize'] = 18
rcParams['legend.fontsize'] = 18
rcParams['figure.titlesize'] = 18
rcParams['markers.fillstyle'] = 'none'


################################################################################################
#### Aggregate the results                                                                  ####
################################################################################################
RESULT_DIR = 'Fewshot/EDP_self_close/'
model_performance = load_ablation_results(RESULT_DIR)
RESULT_DIR = 'Fewshot/EDP_self_open/'
model_performance += load_ablation_results(RESULT_DIR)
RESULT_DIR = 'Fewshot/BSP_self_close/'
model_performance += load_ablation_results(RESULT_DIR)
RESULT_DIR = 'Fewshot/BSP_self_open/'
model_performance += load_ablation_results(RESULT_DIR)

result_df = []
for expr_result in model_performance:
    result = calculate_accuracy(expr_result)
    expr_df = pd.DataFrame(
        columns=['model', 'problem', 'level', 'Average accuracy', 'Failure', 'difference'])
    model_name = result['model']
    difference = result['difference']
    problem_name = result['problem']
    accuracy = result['accuracy']
    failed = result['failed']
    accuracy_len = len(accuracy)
    expr_df['model'] = [model_name] * accuracy_len
    expr_df['problem'] = [problem_name] * accuracy_len
    expr_df['level'] = [i+1 for i in range(10 - accuracy_len, 10)]
    expr_df['Average accuracy'] = accuracy
    norm_sum = (21 - accuracy_len) * accuracy_len / 2
    expr_df['weighted_accuracy'] = [
        x * (i + 1) / norm_sum for i, x in zip(range(10 - accuracy_len, 10), accuracy)
    ]
    expr_df['Failure'] = failed
    expr_df['weighted_failed'] = [x / accuracy_len for i, x in enumerate(failed)]
    expr_df['difference'] = [difference] * accuracy_len
    expr_df['lvl_correctness'] = result['level_correctness']
    expr_df['is_close'] = [model_name in close_models] * accuracy_len
    result_df.append(expr_df)
result_df = pd.concat(result_df)
result_df['model'] = result_df['model'].map(model_mapper)
result_df.to_csv('result_fewshot.csv')

model_names = [None for _ in range(len(model_performace))]
for model, idx in model_performace.items():
    model_names[idx - 1] = model
print(model_names)

for i, problem in enumerate(['bspResults', 'edpResults']):
    fig, _ = plt.subplots(nrows=6, ncols=2)
    for model in result_df.model.unique():
        tmp_df = result_df[(result_df.model == model) & (result_df.problem == problem)]
        tmp_df = tmp_df.sort_values(by=['difference', 'level'])
        tmp_df = tmp_df.pivot(
            index='difference',
            columns='level',
            values='Average accuracy'
        ).sort_values(by='difference', ascending=False)
        plt.subplot(6, 2, model_names.index(model) + 1)
        sns.heatmap(tmp_df, annot=True, vmin=0, vmax=1, cmap='Reds', fmt='.2f', cbar=False)
        plt.title(model)
        plt.xlabel(None)
        plt.ylabel('Difficulty Difference')
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.97, 0.68, 0.02, 0.2])
    fig.colorbar(ax=fig.axes, cax=cbar_ax,
                    mappable=fig.axes[0].collections[0], orientation='vertical')
    # plt.savefig(f'figures/ablation/{problem}_accuracy.png', bbox_inches='tight')
    plt.savefig(f'figures/ablation/{problem}_accuracy.png', bbox_inches='tight')
