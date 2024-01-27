"""Model performance on different complexity problems."""
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd

from visualize_utils import *

################################################################################################
#### Plot setting                                                                           ####
################################################################################################

rcParams['figure.dpi'] = 500
rcParams['savefig.dpi'] = 500
rcParams['figure.figsize'] = 15, 6
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 18
rcParams['axes.labelsize'] = 18
rcParams['axes.titlesize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 18
rcParams['figure.titlesize'] = 18
rcParams['markers.fillstyle'] = 'none'

################################################################################################
#### Load and aggregate the results                                                         ####
################################################################################################
RESULT_DIR = 'Zeroshot'
EXTRA_RESULT_DIR = 'Zeroshot_extra'
model_performance = load_results(RESULT_DIR) + load_results(EXTRA_RESULT_DIR)
result_df = []
for expr_result in model_performance:
    if expr_result is None:
        continue
    result = calculate_accuracy(expr_result)
    expr_df = pd.DataFrame(columns=['model', 'problem', 'level', 'Average accuracy', 'Failure'])
    model_name = result['model']
    problem_name = result['problem']
    accuracy = result['accuracy']
    failed = result['failed']
    expr_df['model'] = [model_name] * 10
    expr_df['problem'] = [problem_name] * 10
    expr_df['level'] = [f'Lvl {i+1}' for i in range(10)]
    expr_df['Average accuracy'] = accuracy
    expr_df['weighted_accuracy'] = [x * (i+1) / 55 for i, x in enumerate(accuracy)]
    expr_df['Failure'] = failed
    expr_df['weighted_failed'] = [x / 10 for i, x in enumerate(failed)]
    expr_df['complexity'] = [problem_mapper[problem_name]] * 10
    expr_df['lvl_correctness'] = result['level_correctness']
    expr_df['is_close'] = [model_name in close_models] * 10
    result_df.append(expr_df)
result_df = pd.concat(result_df)

result_df['complexity'] = result_df['complexity'].apply(lambda x: complexity_mapper[x])
result_df['model'] = result_df['model'].map(model_mapper)
result_df.to_csv('results.csv', index=False)

result_df = result_df[result_df['problem'] != 'mfpResults']
tmp_df = result_df.groupby(['model', 'problem', 'complexity', 'is_close'], as_index=False).agg({
    'Average accuracy': 'mean',
    'weighted_accuracy': 'sum', 
    'weighted_failed': 'sum'
}).reset_index()


################################################################################################
#### Plot the results                                                                       ####
################################################################################################
# Change the column name to visualize different metrics
# col_name = 'Average accuracy'
# col_name = 'weighted_accuracy'
# col_name = 'weighted_failed'

def plot_final_output(col_name, df):
    """Plot the final output."""
    tmp_df = df.copy()
    tmp_df = tmp_df.groupby(
        ['model', 'complexity', 'is_close'],
        as_index=False
    ).agg({col_name: 'mean'}).reset_index()
    mean_tmp_df = tmp_df.groupby(['complexity'], as_index=False).agg({col_name: 'mean'})
    mean_tmp_df['comp_order'] = mean_tmp_df['complexity'].map(comp_level)
    tmp_df['comp_order'] = tmp_df['complexity'].map(comp_level)
    tmp_df.sort_values(by=['comp_order', 'is_close'], inplace=True)
    mean_tmp_df.sort_values(by=['comp_order'], inplace=True)

    open_model_df = tmp_df[~tmp_df['is_close']]
    close_model_df = tmp_df[tmp_df['is_close']]
    number_of_close_models = close_model_df['model'].nunique()
    number_of_open_models = open_model_df['model'].nunique()

    # make one red palette and one blue palette
    palette = sns.color_palette('tab20', n_colors=number_of_close_models + number_of_open_models)
    palette = sorted(palette, key=lambda x: x[0] - x[2])
    palette_map = {}

    sns.pointplot(data=tmp_df[~tmp_df['is_close']], x='complexity',
            y=col_name, hue='model', linestyle='',
            alpha=0.8, marker='s', palette=palette[:number_of_open_models])
    palette_map = {model: palette[i] for i, model in enumerate(open_model_df['model'].unique())}

    sns.pointplot(data=tmp_df[tmp_df['is_close']], x='complexity', y=col_name,
                    hue='model', linestyle='', alpha=0.8, marker='^', palette=palette[number_of_open_models:])
    palette_map.update({model: palette[i + number_of_open_models] for i, model in enumerate(close_model_df['model'].unique())})

    sns.lineplot(data=tmp_df, x='complexity', y=col_name, color='black',
                    marker='o', markersize=10, fillstyle='full', label='All models', errorbar=None)

    for model in ['GPT 4 Turbo', 'Claude 2', 'Phi-2', 'Mistral-7b']:
        sns.lineplot(data=tmp_df[tmp_df['model'] == model], x='complexity', y=col_name, color=palette_map[model],
                    linestyle='-.',  label=None, errorbar=None, alpha=0.8)
    for _, row in mean_tmp_df.iterrows():
        plt.text(row['comp_order'] - 1, row[col_name] + 0.02, f'{row[col_name]:.2f}', color='black')

    # set the title and labels
    if col_name == 'weighted_accuracy':
        plt.title("a.", loc='left')
        plt.ylabel('Weighted accuracy')
    elif col_name == 'Average accuracy':
        plt.ylabel('Average accuracy')
    else:
        plt.title("b.", loc='left')
        plt.ylabel('Weighted failure rate')
    # close the legend
    plt.xlabel('Complexity')
    if col_name != 'weighted_failed':
        plt.legend().remove()
    else:
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.ylim(0, 1.05)


fig, _ = plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plot_final_output('weighted_accuracy', tmp_df)
plt.subplot(1, 2, 2)
plot_final_output('weighted_failed', tmp_df)
plt.savefig('figures/rq1/weighted_accuracy_failed.png', bbox_inches='tight')
