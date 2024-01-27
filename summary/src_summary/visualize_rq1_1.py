"""Zero-shot performance of the models on the 9 problems."""
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from visualize_utils import *


################################################################################################
#### Basic settings for the plots                                                           ####
################################################################################################
rcParams['figure.dpi'] = 500
rcParams['savefig.dpi'] = 500
rcParams['figure.figsize'] = 15, 16
rcParams['font.family'] = 'Arial'
rcParams['axes.labelsize'] = 18
rcParams['axes.titlesize'] = 18
rcParams['legend.fontsize'] = 18
rcParams['figure.titlesize'] = 18
rcParams['markers.fillstyle'] = 'none'


################################################################################################
#### Data pre-processing                                                                    ####
################################################################################################

result_df = pd.read_csv('results.csv')
result_df = result_df[result_df['problem'] != 'mfpResults']
result_df['level'] = result_df['level'].apply(lambda x: x.split(' ')[1])
result_df['level'] = result_df['level'].astype(int)
result_df["comp_level"] = result_df["complexity"].map(comp_level)
problem_name = result_df.sort_values(
    by=['comp_level', 'problem'],
    ascending=[False, True]
).problem.unique().tolist()
problem_name = [
    'gcpResults',
    'tspResults',
    'mspResults',
    'gcp_d_Results',
    'tsp_d_Results',
    'kspResults',
    'bspResults',
    'edpResults',
    'sppResults'
]


################################################################################################
#### Plot the figure                                                                        ####
################################################################################################

fig, _ = plt.subplots(3, 3, figsize=(15, 12))
for problem in result_df.problem.unique():
    tmp_df = result_df[(result_df.problem == problem)]
    tmp_df = tmp_df.pivot(index='model', columns='level', values='Average accuracy').reset_index()
    tmp_df.sort_values(by='model', inplace=True, key=lambda x: x.map(model_performace))
    tmp_df.set_index('model', inplace=True)
    # plt.figure(figsize=(10, 10))
    pos = problem_name.index(problem) + 1
    plt.subplot(3, 3, pos)
    sns.heatmap(tmp_df, annot=True, vmin=0, vmax=1, cmap='Blues', fmt='.2f', cbar= False)
    problem = problem[:-8] if problem.endswith('_Results') else problem[:-7]
    plt.title(problem.upper() if problem != 'bsp' else 'SAS')
    plt.xlabel(None)
    plt.ylabel(None)
    plt.tight_layout()

# draw the colorbar
fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.96, 0.7, 0.02, 0.27])
fig.colorbar(ax=fig.axes, cax=cbar_ax, mappable=fig.axes[0].collections[0], orientation='vertical')
plt.savefig('figures/rq1/zeroshot_heatmap.png', bbox_inches='tight')
