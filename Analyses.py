import numpy as np
import seaborn as sns
import pandas as pd
import json
# from replay_python3 import *
from model import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats
import copy


import math


### Global variables ###

BOARD_NAMES = ['6_hard_full','6_hard_pruned','10_hard_full','10_hard_pruned', '6_easy_full','6_easy_pruned','10_easy_full','10_easy_pruned', '10_medium_full','10_medium_pruned']
# these are the boards starting positions (1 = X, 2 = O)
START_POSITION = [[[0,2,0,0,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[0,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                  [[0,2,0,1,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[2,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                  [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,0,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                 [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,1,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[2,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                  [[0,1,0,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,0],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                  [[0,1,2,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,1],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                [[0,0,0,0,1,0,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                  [[0,0,0,0,1,2,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,1],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]],
                 [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,1,1,1,2,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]]
                  ]

START_POSITIONS_DICT = {'6_hard_full': [[0,2,0,0,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[0,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                        '6_hard_pruned': [[0,2,0,1,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[2,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                        '10_hard_full': [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,0,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                        '10_hard_pruned': [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,1,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[2,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                        '6_easy_full': [[0,1,0,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,0],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                        '6_easy_pruned':  [[0,1,2,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,1],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                        '10_easy_full':[[0,0,0,0,1,0,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                        '10_easy_pruned':  [[0,0,0,0,1,2,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,1],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                        '10_medium_full': [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]],
                        '10_medium_pruned': [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,1,1,1,2,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]]
                  }



BOARD_NAMES = {'6_hard_full': 'Board II full','6_hard_pruned': 'Board II truncated','10_hard_full': 'Board IV full',
               '10_hard_pruned': 'Board IV truncated', '6_easy_full':'Board I full','6_easy_pruned': 'Board I truncated',
               '10_easy_full': 'Board III full','10_easy_pruned': 'Board III truncated',
               '10_medium_full': 'Board V full','10_medium_pruned': 'Board V truncated'}



### Global variables end ###



### Utility methods ###

def read_matrices_from_file(filename):
  json1_file = open(filename)
  json1_str = json1_file.read()
  json1_data = json.loads(json1_str)
  return json1_data


def rank_biserial_effect_size(x,y):
    mann_whitney_res = stats.mannwhitneyu(x, y)
    print(mann_whitney_res)
    u = mann_whitney_res[0]
    effect_size = 1.0-((2.0*u)/(len(x)*len(y)))
    return effect_size


def bootstrap_mean(x, B=10000, alpha=0.05, plot=False):
    """Bootstrap standard error and (1-alpha)*100% c.i. for the population mean

    Returns bootstrapped standard error and different types of confidence intervals"""

    # Deterministic things
    n = len(x)  # sample size
    orig = x.mean()  # sample mean
    se_mean = x.std()/np.sqrt(n) # standard error of the mean
    qt = stats.t.ppf(q=1 - alpha/2, df=n - 1) # Student quantile

    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = xboot.mean(axis=1)

   # Standard error and sample quantiles
    se_mean_boot = sampling_distribution.std()
    quantile_boot = np.percentile(sampling_distribution, q=(100*alpha/2, 100*(1-alpha/2)))

    # # RESULTS
    # print("Estimated mean:", orig)
    # print("Classic standard error:", se_mean)
    # print("Classic student c.i.:", orig + np.array([-qt, qt])*se_mean)
    # print("\nBootstrap results:")
    # print("Standard error:", se_mean_boot)
    # print("t-type c.i.:", orig + np.array([-qt, qt])*se_mean_boot)
    # print("Percentile c.i.:", quantile_boot)
    # print("Basic c.i.:", 2*orig - quantile_boot[::-1])

    if plot:
        plt.hist(sampling_distribution, bins="fd")
    # return sampling_distribution
    return np.round(orig, decimals=2), np.round(quantile_boot, decimals=2)


def spearmanr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    # r, p = stats.pearsonr(x,y)
    # r_z = np.arctanh(r)
    # se = 1/np.sqrt(x.size-3)
    # z = stats.norm.ppf(1-alpha/2)
    # lo_z, hi_z = r_z-z*se, r_z+z*se
    # lo, hi = np.tanh((lo_z, hi_z))
    r, p = stats.spearmanr(x,y)
    stderr = 1.0 / math.sqrt(x.size - 3)
    z = stats.norm.ppf(1-alpha/2)
    # print(z)
    delta = z * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)
    # print "lower %.6f upper %.6f" % (lower, upper)
    return r, p, lower, upper


def bootstrap_median(x, B=10000, alpha=0.05, plot=False):
    """Bootstrap standard error and (1-alpha)*100% c.i. for the population mean

    Returns bootstrapped standard error and different types of confidence intervals"""

    x = x[~np.isnan(x)]
    # Deterministic things
    n = len(x)  # sample size
    orig = np.median(x)  # sample median
    se_mean = x.std()/np.sqrt(n) # standard error of the mean
    qt = stats.t.ppf(q=1 - alpha/2, df=n - 1) # Student quantile

    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    # sampling_distribution = xboot.median(axis=1)

    sampling_distribution = np.median(xboot, axis=1)

   # Standard error and sample quantiles
    se_mean_boot = sampling_distribution.std()
    quantile_boot = np.percentile(sampling_distribution, q=(100*alpha/2, 100*(1-alpha/2)))

    if plot:
        plt.hist(sampling_distribution, bins="fd")
    return np.round(orig, decimals=2), np.round(quantile_boot, decimals=2)


def boot_matrix(z, B):
    """Bootstrap sample

    Returns all bootstrap samples in a matrix"""

    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]


def bootstrap_mean_diffs(data, group_column, group_names, alpha=0.05, num_samples=1000):
    diff_samples = np.zeros(num_samples)
    for i in range(num_samples):
        data_sample = data.sample(frac=1, replace=True)
        diff = data_sample[data[group_column] == group_names[0]]['action'].mean()- data_sample[data[group_column] == group_names[1]]['action'].mean()
        diff_samples[i] = diff

    quantile_boot = np.percentile(diff_samples, q=(100*alpha/2, 100*(1-alpha/2)))
    return quantile_boot


def bootstrap_ci_for_heuristics_vs_people(dist_df, heuristic1, heuristic2, ci=0.05):
    num_simulations = dist_df.sample_number.max()
    mean_diffs = []
    for i in range(num_simulations):
        mean_h1 = dist_df.loc[(dist_df['sample_number'] == i) & (dist_df['heuristic'] == heuristic1)].distance_normalized_chance.mean()
        mean_h2 = dist_df.loc[(dist_df['sample_number'] == i) & (dist_df['heuristic'] == heuristic2)].distance_normalized_chance.mean()
        mean_diffs.append(mean_h1-mean_h2)

    return np.percentile(np.array(mean_diffs), q=(100*ci/2, 100*(1-ci/2)))

### Utility methods end ###


### Methods for generating figures and statistical analyses ###


def search_size_correctness_figure(alpha_beta_df, behavioral_df):
    '''figure 2 - participants' search size vs. complexity and participants' correctness vs. complexity'''

    # generate behavioral data with action counts
    behavioral_clicks_df = behavioral_df.loc[((behavioral_df['action']=='click'))]

    actions_counts_df = behavioral_clicks_df.groupby(['userid','solved','board_name', 'board_size','board_type','condition'], as_index=False)['action'].count()

    # correlation alpha-beta interaction and participants moves
    actions_counts__board_means_df = actions_counts_df.groupby(['board_name', 'board_size','board_type','condition'], as_index=False)['action'].mean()
    actions_counts__board_means_df['type'] = actions_counts__board_means_df['board_type'].map({'hard': 'HC', 'easy': 'MC', 'medium': 'DC'})
    actions_counts__board_means_df['actionsSolution'] = actions_counts__board_means_df['action']
    actions_counts__board_means_df['size'] = actions_counts__board_means_df['board_size']
    actions_counts__board_means_df['condition'] = actions_counts__board_means_df['condition'].map({'full': 'full', 'pruned': 'truncated'})

    alpha_beta_df['heuristic_name'] = alpha_beta_df['heuristic_name'].map({'density':'density', 'linear':  'linear','non-linear':'non-linear', 'non-linear-interaction': 'interaction','blocking':'blocking', 'participants':'participants'})

    # board complexity is determined according to the "interaction" heuristic
    alphaBetaInteraction = alpha_beta_df.loc[alpha_beta_df['heuristic_name'] == 'interaction']

    # preprocessing
    behavioral_agg_df = behavioral_clicks_df.groupby(['userid','board_size','board_type','condition','board_name','solved']).action.count().reset_index()
    behavioral_agg_df['solutionAndValidationCorrectPercent'] = behavioral_agg_df['solved'].map({'wrong': 0, 'solvedCorrect':0, 'validatedCorrect': 100})
    behavioral_agg_df['solutionAndValidationCorrect'] = behavioral_agg_df['solved'].map({'wrong': 0, 'solvedCorrect':0, 'validatedCorrect': 1})
    behavioral_agg_df['solutionCorrect'] = behavioral_agg_df['solved'].map({'wrong': 0, 'solvedCorrect':1, 'validatedCorrect': 1})

    behavioral_agg_df['type'] = behavioral_agg_df['board_type'].map({'hard': 'HC', 'easy': 'MC', 'medium': 'DC'})
    behavioral_agg_df['actionsSolution'] = behavioral_agg_df['action']
    behavioral_agg_df['size'] = behavioral_agg_df['board_size']
    behavioral_agg_df['condition'] = behavioral_agg_df['condition'].map({'full': 'full', 'pruned': 'truncated'})

    # prints mean and median number of actions for each board
    print('mean and median number of actions for each board: \n')
    for board_name in BOARD_NAMES:
        action_counts_board = actions_counts_df.loc[actions_counts_df['board_name'] == board_name]
        print(board_name)
        print('median', action_counts_board['action'].median())
        print('mean', action_counts_board['action'].mean())

    print('-------------------------------------------------------------------------')

    alpha_beta_behavioral_df = pd.merge(alphaBetaInteraction, behavioral_agg_df, on=['size','type','condition'], how='left')

    alpha_beta_behavioral__interaction_df = alpha_beta_behavioral_df.loc[alpha_beta_behavioral_df['heuristic_name'] == 'interaction']

    print(' spearman correlation between number of user actions and number of solution moves by alpha beta:')
    print(spearmanr_ci(alpha_beta_behavioral__interaction_df.action.values, alpha_beta_behavioral__interaction_df.moves.values))
    print(' spearman correlation between correctness of users and number of solution moves by alpha beta:')
    print(spearmanr_ci(alpha_beta_behavioral__interaction_df.solutionAndValidationCorrect.values, alpha_beta_behavioral__interaction_df.moves.values))

    print('-------------------------------------------------------------------------')

    # make plot for Figure 2 in the paper
    plt.subplots(1,2, figsize=(8.6,4))
    plt.subplot(1, 2, 2)
    ax = sns.regplot(x="moves", y="solutionAndValidationCorrectPercent", x_estimator=np.mean, data=alpha_beta_behavioral_df, color="r", fit_reg=False,  ci=68)
    ax.set(xscale="log")
    ax.set(ylim=(0, 100))
    ax.set(xlim=(10, 200000))
    ax.set_xlabel('Board Complexity', fontsize=14)
    ax.set_ylabel('Percent Correct', fontsize=14)
    ax.tick_params(labelsize=12)
    fig = plt.subplot(1, 2, 1)

    ax = sns.regplot(x="moves", y="actionsSolution", data=alpha_beta_behavioral_df,  x_estimator=np.mean, ci=68, color="b", fit_reg=False)
    sns.regplot(x="moves", y="moves", data=alpha_beta_behavioral_df,  x_estimator=np.mean, ci=68, color="silver", fit_reg=False, ax=ax)

    ax.set(xscale="log")
    ax.set(yscale="log")
    ax.set(xlim=(10, 200000))
    ax.set_xlabel('Board Complexity', fontsize=14)
    ax.set_ylabel('Search Size', fontsize=14)
    ax.tick_params(labelsize=12)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Participants Search Size',
                          markerfacecolor='b', markersize=9),
                       Line2D([0], [0], marker='o', color='w', label='Alpha-Beta Search Size',
                          markerfacecolor='silver', markersize=9)]
    fig.legend(handles=legend_elements, fontsize=11)

    ax.set(xlabel='Board Complexity', ylabel='Search Size')
    plt.tight_layout(pad=1)
    plt.show()

    # search size and correctness tests (data in SI)
    print('min actions', alpha_beta_behavioral__interaction_df.actionsSolution.min())
    correct_participants = alpha_beta_behavioral__interaction_df.loc[alpha_beta_behavioral__interaction_df['solutionAndValidationCorrect'] == 1]
    wrong_participants = alpha_beta_behavioral__interaction_df.loc[alpha_beta_behavioral__interaction_df['solutionAndValidationCorrect'] == 0]
    print('mean correct', correct_participants.actionsSolution.mean())
    print('mean wrong', wrong_participants.actionsSolution.mean())
    print('median correct', correct_participants.actionsSolution.median())
    print('median wrong', wrong_participants.actionsSolution.median())
    print(stats.mannwhitneyu( correct_participants.actionsSolution.values, wrong_participants.actionsSolution.values))
    print(rank_biserial_effect_size(correct_participants.actionsSolution.values, wrong_participants.actionsSolution.values))
    print('median correct ci', bootstrap_median(correct_participants.actionsSolution.values))
    print('median correct ci', bootstrap_median(wrong_participants.actionsSolution.values))

    for board_name in BOARD_NAMES:
        action_counts_board_correct= correct_participants.loc[correct_participants['board_name'] == board_name]
        action_counts_board_wrong= wrong_participants.loc[wrong_participants['board_name'] == board_name]
        print('-------------------------------------')
        print(board_name)
        print('mean correct', action_counts_board_correct['actionsSolution'].mean())
        print('mean wrong', action_counts_board_wrong['actionsSolution'].mean())
        print('mean diff', action_counts_board_wrong['actionsSolution'].mean()-action_counts_board_correct['actionsSolution'].mean())
        print('-------------------------------------')



def generate_participants_heuristics_heatmpaps(participants_data_file, heuristic):

    participants_data = read_matrices_from_file(participants_data_file)
    for board_name in participants_data.keys():
        participants_mat = participants_data[board_name]
        scores = compute_heuristic_scores_board(START_POSITIONS_DICT[board_name], 1, heuristic, True)
        annot_mat = copy.deepcopy(scores)
        for i in range(len(scores)):
            for j in range(len(scores)):
                if scores[i][j] == -0.00001:
                    annot_mat[i][j] = 'X'
                elif scores[i][j] == -0.00002:
                    annot_mat[i][j] = 'O'
                else:
                    annot_mat[i][j] = np.round(scores[i][j], 2)

        annot_participants_mat = copy.deepcopy(scores)
        for i in range(len(participants_mat)):
            for j in range(len(participants_mat)):
                if participants_mat[i][j] == -0.00001:
                    annot_participants_mat[i][j] = 'X'
                elif participants_mat[i][j] == -0.00002:
                    annot_participants_mat[i][j] = 'O'
                else:
                    annot_participants_mat[i][j] = np.round(participants_mat[i][j], 2)

        plt.subplot(2, 1, 2)
        ax = sns.heatmap(scores, annot=annot_mat, fmt='', cmap='rocket_r', yticklabels=False, xticklabels=False, vmin=0,
                         vmax=0.5, square=True)
        plt.suptitle(BOARD_NAMES[board_name])
        plt.title(heuristic.capitalize())
        plt.subplot(2, 1, 1)
        ax = sns.heatmap(participants_mat, annot=annot_participants_mat, fmt='', cmap='rocket_r', yticklabels=False,
                         xticklabels=False, vmin=0,
                         vmax=0.5, square=True)
        plt.title("Participants First Moves")
        plt.show()



def make_emd_correlations_fig(with_mcts=False):
    '''Generates figure 3B: normalized distances between each heuristic and participants' first moves distribution'''

    # read datasets with computed distances between participants and heuristics/mcts
    dist_df_bs = pd.read_csv('stats/emd_heuristics_participants_to_chance.csv')
    dist_df_bs['heuristic'] = dist_df_bs['heuristic'].map({'mcts':'mcts','density':'Density', 'linear':  'Linear',
                                                     'non-linear':'Non-linear', 'interaction': 'Interaction',
                                                     'forcing':'Forcing', 'mcts_k5_100_norollouts': 'MCTS k=5\n n=100',
                                                     'mcts_k5_5000_norollouts': 'MCTS k=5\n n=5000',
                                                     'mcts_all_100_noRollouts': 'MCTS All\n n=100',
                                                     'mcts_all_5000_norollouts': 'MCTS All\n n=5000'})

    dist_df_by_agg = pd.read_csv('stats/emd_heuristics_mcts_participants_to_chance_agg.csv')
    dist_df_by_agg = dist_df_by_agg.groupby(['heuristic','sample_number'], as_index=False).distance_normalized_chance.mean()

    dist_df_by_agg['heuristic'] = dist_df_by_agg['heuristic'].map({'mcts':'mcts','density':'Density', 'linear':  'Linear',
                                                     'non-linear':'Non-linear', 'interaction': 'Interaction',
                                                     'forcing':'Forcing', 'mcts_k5_100_norollouts': 'MCTS k=5\n n=100',
                                                     'mcts_k5_5000_norollouts': 'MCTS k=5\n n=5000',
                                                     'mcts_all_100_noRollouts': 'MCTS All\n n=100',
                                                     'mcts_all_5000_norollouts': 'MCTS All\n n=5000'})

    heuristics = ['Density','Linear','Non-linear','Interaction']

    if with_mcts:
        heuristics = ['MCTS All\n n=100', 'MCTS All\n n=5000', 'MCTS k=5\n n=100', 'Density', 'Interaction',
                      'MCTS k=5\n n=5000']

    # confidence intervals for distances between heuristics and participants
    confidence_level = 0.95
    print("Confidence intervals for the normalized earth mover distances between participants and the heuristics, confidence level = " + str(confidence_level))
    for heuristic in heuristics:
        print(heuristic)
        heuristic_df = dist_df_by_agg.loc[(dist_df_by_agg.heuristic == heuristic)]
        print(np.percentile(heuristic_df.distance_normalized_chance.values, q=(100*(confidence_level)/2, 100*((1-confidence_level)/2))))


    dist_df = pd.read_csv('stats/emd_heuristics_participants_to_chance_raw_first_only.csv')
    dist_df['heuristic'] = dist_df['heuristic'].map({'mcts':'mcts','density':'Density', 'linear':  'Linear',
                                                     'non-linear':'Non-linear', 'interaction': 'Interaction',
                                                     'forcing':'Forcing', 'mcts_k5_100_norollouts': 'MCTS k=5\n n=100',
                                                     'mcts_k5_5000_norollouts': 'MCTS k=5\n n=5000',
                                                     'mcts_all_100_noRollouts': 'MCTS All\n n=100',
                                                     'mcts_all_5000_norollouts': 'MCTS All\n n=5000'})

    # statistical tests for for differences between the distances of each pair of heuristics from participants
    heuristics_to_compare = ['Density', 'Linear', 'Non-linear', 'Interaction']
    print("Statistical tests for differences between heuristics:")
    for i in range(len(heuristics_to_compare)-1):
        print('------------------------------------')
        for j in range(i+1,len(heuristics_to_compare)):
            print('comparing:', heuristics_to_compare[i] + ' to ' + heuristics_to_compare[j])
            heuristics_i_df = dist_df.loc[(dist_df.heuristic == heuristics_to_compare[i])]
            heuristics_j_df = dist_df.loc[(dist_df.heuristic == heuristics_to_compare[j])]
            # bootstrap confidence interval for the difference between the mean heuristic distances
            print('bootstrap test', bootstrap_ci_for_heuristics_vs_people(dist_df_bs,heuristics_to_compare[i],heuristics_to_compare[j],ci=1-confidence_level))
            print('mann whitney test', rank_biserial_effect_size(heuristics_i_df.distance_normalized_chance.values,heuristics_j_df.distance_normalized_chance.values))


    # make figure 3B - barplot of normalized distances between each heuristic and participants' first moves distribution
    plt.figure(figsize=(3, 3))

    plt.title('Wasserstein distance between first move distributions predicted by the different scoring strategies and participants first moves')
    ax = sns.barplot(x='heuristic', y='distance_normalized_chance',ci='boot_custom', order=heuristics, data=dist_df_by_agg)
    ax.set_xlabel('')
    ax.set_ylabel('Distance (normalized)', fontsize=16)
    plt.show()


def generate_correlation_figures():
    # make figure 3D - barplot of the correlations between the predicted likelihood of paths and the observed likelihood of paths according to participants search patterns
    correlations_df = pd.read_csv('stats/heuristics_participants_paths_correlations_lengths_agg_spearman_5_0.01.csv')
    # group by sample
    aggregation = {
        'correlation': 'mean'
    }
    correlations_df_agg = correlations_df.groupby(['heuristic', 'sample_number'], as_index=False).agg(aggregation)
    correlations_df_agg = correlations_df_agg[['heuristic', 'sample_number', 'correlation']]

    path_probs_participants_heuristic_df = pd.read_csv('stats/probs_paths_participants_heuristic.csv')

    # rename heuristics (capital letter, blocking --> forcing)
    correlations_df_agg['heuristic'] = correlations_df_agg['heuristic'].map({'density':'Density', 'linear':  'Linear','non-linear':'Non-linear', 'interaction': 'Interaction','blocking':'Forcing'})

    heuristics = ['Density', 'Linear', 'Non-linear', 'Interaction']

    # statistical tests for differences in correlations between heuristics and participants
    print('Statistical tests for differences in correlations between heuristics and participants:')
    for i in range(len(heuristics)):
        print(heuristics[i])
        heuristic_i_data = correlations_df_agg.loc[correlations_df_agg.heuristic == heuristics[i]]
        for j in range(i+1, len(heuristics)):
            print(heuristics[j])
            heuristic_j_data = correlations_df_agg.loc[correlations_df_agg.heuristic == heuristics[j]]
            print(stats.mannwhitneyu(heuristic_i_data.correlation.values, heuristic_j_data.correlation.values, alternative='less'))

    # figure 3D - Correlations between the predicted likelihood of paths and the observed likelihood of paths according to participants search patterns
    plt.figure(figsize=(3, 3))
    ax = sns.barplot(x='heuristic',y='correlation', order=['Density', 'Linear', 'Non-linear', 'Interaction'], ci='boot_custom', data=correlations_df_agg)  # uses custom change to seaborn to handle bootstrap confidence interval
    ax.set_xlabel('')
    ax.set_ylabel('Correlation', fontsize=16)
    plt.show()


    # figure 3C - Path likelihoods predicted by the ``Interaction'' scoring strategy vs. observed path likelihoods for paths of length smaller than 4
    sns.set(font_scale=2.5, style="whitegrid")
    # filter relevant data (interaction, path length <4)
    path_probs_participants_heuristic_df = path_probs_participants_heuristic_df.loc[(path_probs_participants_heuristic_df['heuristic'] == 'interaction') & (path_probs_participants_heuristic_df['move_number_in_path'] <4) & (path_probs_participants_heuristic_df['participants_prob'] >= 0.05)]
    ax = sns.lmplot(x='participants_prob', y='prob_path', data=path_probs_participants_heuristic_df, fit_reg=True, scatter_kws={"s": 45})
    ax = ax.set_axis_labels('Observed Likelihood', 'Predicted Likelihood')

    plt.show()


def plot_entropies_full_truncated(filename, solvers_filename=None, nonsolvers_filename=None):
    '''Figure 4C, S9 and S10 - entropies full vs. truncated boards'''

    entropies_data = pd.read_csv(filename)
    mean_entropy_full = entropies_data.loc[entropies_data['condition'] == 'Full']
    mean_entropy_full = mean_entropy_full['entropy'].values
    mean_entropy_pruned = entropies_data.loc[entropies_data['condition'] == 'Truncated']
    mean_entropy_pruned = mean_entropy_pruned['entropy'].values
    full_ci = np.nanpercentile( np.asarray(mean_entropy_full), q=(100*0.05/2, 100*(1-0.05/2))) # confidence interval
    pruned_ci = np.nanpercentile( np.asarray(mean_entropy_pruned), q=(100*0.05/2, 100*(1-0.05/2))) # confidence interval

    print('--stats all--')
    print(np.mean(mean_entropy_full))
    print(full_ci)
    print(np.mean(mean_entropy_pruned))
    print(pruned_ci)
    print(stats.mannwhitneyu(mean_entropy_full, mean_entropy_pruned))
    print(rank_biserial_effect_size(mean_entropy_full, mean_entropy_pruned))

    # plot entropies full vs. truncated
    plt.figure(figsize=(5,5))
    ax = sns.barplot(x='condition', y='entropy',ci='boot_custom', data=entropies_data)
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('Entropy', fontsize=14)
    plt.show()


    # plot entropies full vs. truncated seperately for solvers and non-solvers
    if (solvers_filename is not None) & (nonsolvers_filename is not None):
        solvers_data = pd.read_csv(solvers_filename)
        nonsolvers_data = pd.read_csv(nonsolvers_filename)

        plt.subplot(1,2,1)
        ax = sns.barplot(x='condition', y='entropy',ci='boot_custom', data=solvers_data)
        ax.set_xlabel('Condition', fontsize=14)
        ax.set_ylabel('Entropy (solvers)', fontsize=14)

        ax.tick_params(labelsize=14)

        plt.subplot(1,2,2)
        ax = sns.barplot(x='condition', y='entropy',ci='boot_custom', data=nonsolvers_data)
        ax.set_xlabel('Condition', fontsize=14)
        ax.set_ylabel('Entropy (non-solvers)', fontsize=14)
        ax.tick_params(labelsize=14)
        plt.tight_layout(pad=1.5)
        plt.show()

        mean_entropy_full = solvers_data.loc[solvers_data['condition'] == 'Full']
        mean_entropy_full = mean_entropy_full['entropy'].values
        mean_entropy_pruned = solvers_data.loc[solvers_data['condition'] == 'Truncated']
        mean_entropy_pruned = mean_entropy_pruned['entropy'].values
        full_ci = np.nanpercentile( np.asarray(mean_entropy_full), q=(100*0.05/2, 100*(1-0.05/2)))
        pruned_ci = np.nanpercentile( np.asarray(mean_entropy_pruned), q=(100*0.05/2, 100*(1-0.05/2)))

        print('--stats solvers--')
        print(np.mean(mean_entropy_full))
        print(full_ci)
        print(np.mean(mean_entropy_pruned))
        print(pruned_ci)
        print(stats.mannwhitneyu(mean_entropy_full, mean_entropy_pruned))
        print(rank_biserial_effect_size(mean_entropy_full, mean_entropy_pruned))

        mean_entropy_full = nonsolvers_data.loc[nonsolvers_data['condition'] == 'Full']
        mean_entropy_full = mean_entropy_full['entropy'].values
        mean_entropy_pruned = nonsolvers_data.loc[nonsolvers_data['condition'] == 'Truncated']
        mean_entropy_pruned = mean_entropy_pruned['entropy'].values
        full_ci = np.nanpercentile( np.asarray(mean_entropy_full), q=(100*0.05/2, 100*(1-0.05/2)))
        pruned_ci = np.nanpercentile( np.asarray(mean_entropy_pruned), q=(100*0.05/2, 100*(1-0.05/2)))

        print('--stats non-solvers--')
        print(np.mean(mean_entropy_full))
        print(full_ci)
        print(np.mean(mean_entropy_pruned))
        print(pruned_ci)
        print(stats.mannwhitneyu(mean_entropy_full, mean_entropy_pruned))
        print(rank_biserial_effect_size(mean_entropy_full, mean_entropy_pruned))


def generate_entropy_heatmap(dynamics, board_name_base):
    '''figures 4A and 4B - heatmaps showing distirubiotns of first moves in full board and in truncated board'''
    board_name_truncated = board_name_base + '_pruned'
    board_name_full = board_name_base + '_full'
    dynamics = dynamics.loc[dynamics['action'] == 'click']
    # filter data - first move in truncated boards is equivalent to 3rd move in full
    data_full = dynamics.loc[(dynamics['board_name'] == board_name_full) & (dynamics['move_number_in_path'] == 3)]
    data_truncated = dynamics.loc[(dynamics['board_name'] == board_name_truncated) & (dynamics['move_number_in_path'] == 1)]


    # filter only to first moves (in truncated board, and equivalent board state in full boards)
    start_pos = data_truncated['board_state'].iloc[0]
    data_full = data_full.loc[data_full['board_state'] == start_pos]
    data_truncated = data_truncated.loc[data_truncated['board_state'] == start_pos]

    moves_full = data_full['position'].unique()
    moves_truncated = data_truncated['position'].unique()

    # preprocessing - need to replace X and O (1 and 2) with negative values so we know to ignore them
    start_pos = np.array(ast.literal_eval(start_pos))
    for i in range(len(start_pos)):
        for j in range(len(start_pos)):
            if start_pos[i][j] == 1:
                start_pos[i][j] = -1
            if start_pos[i][j] == 2:
                start_pos[i][j] = -2

    data_matrix_full = start_pos

    counter = 0
    for move in moves_full:
        cell = move.split('_')
        row_index = int(cell[0])
        col_index = int(cell[1])
        data_move = data_full.loc[data_full['position'] == move]
        data_matrix_full[row_index][col_index] = data_move.shape[0]
        counter += data_move.shape[0]

    heatmap_full = data_matrix_full.astype(float)
    heatmap_full_annot = data_matrix_full.tolist()

    for i in range(len(heatmap_full)):
        for j in range(len(heatmap_full)):
            if data_matrix_full[i][j] == -1:
                heatmap_full[i][j] = 0
                heatmap_full_annot[i][j] = 'X'
            elif data_matrix_full[i][j] == -2:
                heatmap_full[i][j] = 0
                heatmap_full_annot[i][j] = 'O'
            else:
                heatmap_full[i][j] = data_matrix_full[i][j]/counter
                # heatmap_full_annot[i][j] = np.round(data_matrix_full[i][j] / counter, 2)
                heatmap_full_annot[i][j] = ''


    data_matrix_truncated = start_pos

    counter = 0
    for move in moves_truncated:
        cell = move.split('_')
        row_index = int(cell[0])
        col_index = int(cell[1])
        data_move = data_truncated.loc[data_truncated['position'] == move]
        data_matrix_truncated[row_index][col_index] = data_move.shape[0]
        counter += data_move.shape[0]

    heatmap_truncated = data_matrix_truncated.astype(float)
    heatmap_truncated_annot = data_matrix_truncated.tolist()

    # annotate heatmap with X and O locations
    for i in range(len(heatmap_truncated)):
        for j in range(len(heatmap_truncated)):
            if data_matrix_truncated[i][j] == -1:
                heatmap_truncated[i][j] = 0
                heatmap_truncated_annot[i][j] = 'X'
            elif data_matrix_truncated[i][j] == -2:
                heatmap_truncated[i][j] = 0
                heatmap_truncated_annot[i][j] = 'O'
            else:
                heatmap_truncated[i][j] = data_matrix_truncated[i][j]/counter
                heatmap_truncated_annot[i][j] = ''


    plt.subplot(2, 1, 2)

    sns.set(font_scale=2.3)

    ax = sns.heatmap(heatmap_truncated, annot=heatmap_truncated_annot, fmt='', cmap='rocket_r', yticklabels=False, xticklabels=False, vmin=0,
                     vmax=0.5, square=True, annot_kws={"fontsize":20})
    plt.title('Truncated Board')
    plt.subplot(2, 1, 1)
    ax = sns.heatmap(heatmap_full, annot=heatmap_full_annot, fmt='', cmap='rocket_r', yticklabels=False,
                     xticklabels=False, vmin=0,
                     vmax=0.5, square=True, annot_kws={"fontsize":20})
    plt.title("Full Board")
    plt.show()


def make_shutter_blindness_figure(shutter_correctness_df, shutter_blindness_df):
    ''' figure 5A: probability of winning move by participant shutter size'''
    shutter_blindness_df['missed_win_percent'] = shutter_blindness_df['missed_win'].apply(lambda x: x*100)  # change ratio to percent

    # shutter categories
    shutter_correctness_df['mean_shutter_cat'] = shutter_correctness_df['mean_shutter_cat'].map(
            {'narrow\n(0-0.16)':'narrow\n(0-0.1)','medium\n(0.16-0.5)':'medium\n(0.1-0.67)','wide\n(0.5-2.5)':'wide\n(0.67-3.6)'})
    shutter_correctness_df['correct_percent'] = shutter_correctness_df['correct'].apply(lambda x: x*100)
    ax = sns.barplot(x = 'mean_shutter_cat', y = 'correct_percent', n_boot=1000, data=shutter_correctness_df, ci=68)
    ax.set_xlabel('Shutter size', fontsize=12)
    ax.set_ylabel('Prob. winning move [%]', fontsize=12)
    ax.tick_params(labelsize=11)
    plt.show()


def comp_shutter_blindness(comp_shutter_blindness, number_moves=None, noise_signal=0, k=5):
    ''' figure 5B and S13: missed wins by participant shutter size'''
    people_shutter_blindness_columns_df = pd.read_csv('stats/shutter_blindness_avg_both_players_open_participants_columns.csv')

    comp_shutter_all_ratios = comp_shutter_blindness.loc[(comp_shutter_blindness.x_win_opportunities > 0) & (comp_shutter_blindness.o_win_opportunities > 0) & (comp_shutter_blindness.noise_sig == 0) ]
    if number_moves is not None:
        comp_shutter_all_ratios = comp_shutter_all_ratios.loc[comp_shutter_all_ratios.max_moves == number_moves]

    if noise_signal is not None:
        comp_shutter_all_ratios = comp_shutter_all_ratios.loc[comp_shutter_all_ratios.noise_sig == noise_signal]

    if k is not None:
        comp_shutter_all_ratios = comp_shutter_all_ratios.loc[comp_shutter_all_ratios.k == k]

    comp_shutter_all_ratios['ratio_misses_o_x'] = comp_shutter_all_ratios.o_misses/comp_shutter_all_ratios.x_misses
    comp_shutter_all_ratios['ratio_misses_x_o'] = comp_shutter_all_ratios.x_misses/comp_shutter_all_ratios.o_misses


    pivot_ratios = pd.pivot_table(comp_shutter_all_ratios, values=['ratio_misses_x_o'], index='shutter_cat', aggfunc=np.median, fill_value=0)
    print(pivot_ratios)

    print('-------------')
    people_shutter_blindness_columns_df['ratio_misses_o_x'] = people_shutter_blindness_columns_df.o_misses/people_shutter_blindness_columns_df.x_misses
    people_shutter_blindness_columns_df['ratio_misses_x_o'] = people_shutter_blindness_columns_df.x_misses/people_shutter_blindness_columns_df.o_misses
    pivot_ratios_people = pd.pivot_table(people_shutter_blindness_columns_df, values=['ratio_misses_x_o'], index='shutter_cat',  aggfunc=np.median, fill_value=0)
    print(pivot_ratios_people)

    comp_shutter_all_ratios = comp_shutter_all_ratios[['board','shutter_cat','o_misses',	'x_misses', 'ratio_misses_o_x',	'ratio_misses_x_o']]
    comp_shutter_all_ratios['type'] = 'Computational\n Simulations'
    people_shutter_blindness_columns_df = people_shutter_blindness_columns_df[['board' ,'shutter_cat','o_misses',	'x_misses', 'ratio_misses_o_x',	'ratio_misses_x_o']]
    people_shutter_blindness_columns_df['type']= 'Participants'
    dfs = [comp_shutter_all_ratios, people_shutter_blindness_columns_df]
    comp_people_blindness = pd.concat(dfs)

    sns.set(font_scale=1.2, style="whitegrid")
    ax = sns.barplot(x='type', y='ratio_misses_x_o', hue='shutter_cat', estimator=np.median, ci=68, data=comp_people_blindness, hue_order=['narrow','medium','wide'])
    ax.set_ylabel('Median ratio X to O missed wins', fontsize=18)
    ax.set_xlabel('', fontsize=18)
    ax.tick_params(labelsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=9)

    plt.show()


    shutter_cats = ['narrow', 'medium', 'wide']

    print('----------------------------')
    print('mann whitney computational')

    for i in range(len(shutter_cats)):
        for j in range(i+1, len(shutter_cats)):
            print(shutter_cats[i] +' vs. ' + shutter_cats[j])
            comp_shutter_cat_i = comp_shutter_all_ratios.loc[comp_shutter_all_ratios.shutter_cat == shutter_cats[i]]
            comp_shutter_cat_j = comp_shutter_all_ratios.loc[comp_shutter_all_ratios.shutter_cat == shutter_cats[j]]
            print(stats.mannwhitneyu(comp_shutter_cat_i.ratio_misses_x_o.values, comp_shutter_cat_j.ratio_misses_x_o.values))
            print(rank_biserial_effect_size(comp_shutter_cat_i.ratio_misses_x_o.values, comp_shutter_cat_j.ratio_misses_x_o.values))

    print('----------------------------')
    print('mann whitney people')

    for i in range(len(shutter_cats)):
        for j in range(i+1, len(shutter_cats)):
            print(shutter_cats[i] +' vs. ' + shutter_cats[j])
            people_shutter_cat_i = people_shutter_blindness_columns_df.loc[people_shutter_blindness_columns_df.shutter_cat == shutter_cats[i]]
            people_shutter_cat_j = people_shutter_blindness_columns_df.loc[people_shutter_blindness_columns_df.shutter_cat == shutter_cats[j]]
            print(stats.mannwhitneyu(people_shutter_cat_i.ratio_misses_x_o.values, people_shutter_cat_j.ratio_misses_x_o.values))
            print(rank_biserial_effect_size(people_shutter_cat_i.ratio_misses_x_o.values, people_shutter_cat_j.ratio_misses_x_o.values))

    # make figure S13 - Missed winning moves for the ’X’ and ’O’ player in computationalsimulations with a path shutter, and in the behavioral data.
    sns.set(font_scale=1.2, style="whitegrid")
    plt.subplot(1,2,1)
    ax = sns.barplot(x='type', y='x_misses', hue='shutter_cat', estimator=np.median, ci=68, data=comp_people_blindness, hue_order=['narrow','medium','wide'])  # for SI figure
    ax.set_ylabel('Median X missed wins')
    ax.set_xlabel('', fontsize=14)
    plt.legend(loc='best',fontsize=10)


    plt.subplot(1,2,2)
    ax = sns.barplot(x='type', y='o_misses', hue='shutter_cat', estimator=np.median, ci=68, data=comp_people_blindness, hue_order=['narrow','medium','wide'])  # for SI figure
    ax.set_ylabel('Median O missed wins')
    ax.set_xlabel('', fontsize=14)
    plt.legend(loc='best',fontsize=10)
    plt.tight_layout(pad=2)
    plt.show()

def ab_pareto_tradeoff_analysis():
    ''' figure 6B: shutter pareto optimality analysis'''

    alpha_beta_shutter_pareto_noises = pd.read_csv('stats/pareto_noises_interaction_all.csv')
    result = alpha_beta_shutter_pareto_noises.pivot(index='noise', columns='complexity', values='tradeoff')

    plt.figure(figsize=(4,4))
    ax = sns.heatmap(result, annot=False, fmt="g", cmap='coolwarm', square=True)
    ax.set_xlabel('Complexity', fontsize=16)
    ax.set_ylabel('Noise', fontsize=16)
    ax.tick_params(labelsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=60)
    plt.show()




def ab_pareto_tradeoff_example_figs():
    """Figure 6A: example of tradeoff/no tradeoff between computation & correctness when running alpha-beta pruning with shutter"""

    # read results file with alpha-beta runs
    alpha_beta_shutter = pd.read_csv('stats/ab_pareto/ab_pareto_raw_all.csv')  # raw data all runs
    alpha_beta_shutter_pareto = pd.read_csv('stats/ab_pareto_no_board.csv')  # aggregated data

    # load config without tradeoff between computation and correctness
    alpha_beta_no_tradeoff_config = alpha_beta_shutter_pareto.loc[(alpha_beta_shutter_pareto['max_moves']==30) & (alpha_beta_shutter_pareto['noise_sig']==0.5) & (alpha_beta_shutter_pareto['k']==5) & (alpha_beta_shutter_pareto['board']=='6_hard_full')]
    alpha_beta_no_tradeoff_config_all = alpha_beta_shutter.loc[(alpha_beta_shutter['max_moves']==30) & (alpha_beta_shutter['noise_sig']==0.5) & (alpha_beta_shutter['k']==5) & (alpha_beta_shutter['board']=='6_hard_full')]

    # using inverse computation to see the reduction in resources
    alpha_beta_no_tradeoff_config['inverse_computation'] = alpha_beta_no_tradeoff_config['numberOfHeuristicComp'].apply(lambda x: 1-(x/alpha_beta_no_tradeoff_config['numberOfHeuristicComp'].values.max()))
    alpha_beta_no_tradeoff_config_all['inverse_computation'] = alpha_beta_no_tradeoff_config_all['numberOfHeuristicComp'].apply(lambda x: 1.0-(float(x)/alpha_beta_no_tradeoff_config_all['numberOfHeuristicComp'].values.max()))

    # aggregate values by shutter size
    aggregations = {
        'correct': ['mean','sem'],
        'inverse_computation':['mean','sem']
    }

    alpha_beta_shutter_pareto_no_tradeoff = alpha_beta_no_tradeoff_config_all.groupby(['shutter_size']).agg(aggregations).reset_index()

    # generate plot for no tradeoff config - assign color to each shutter
    colors = sns.color_palette("hls", 4)
    x = alpha_beta_shutter_pareto_no_tradeoff[('inverse_computation','mean')]
    y = alpha_beta_shutter_pareto_no_tradeoff[('correct','mean')]
    x_err = alpha_beta_shutter_pareto_no_tradeoff[('inverse_computation','sem')]
    y_err = alpha_beta_shutter_pareto_no_tradeoff[('correct','sem')]

    figure, axs = plt.subplots(2,1,figsize=(4.5,5))


    plt.subplot(2,1,1)
    for i in range(4):
        plt.errorbar(x[i], y[i], xerr=x_err[i], yerr=y_err[i], lw=2, capsize=5, capthick=2, color=colors[i],zorder=1)

    color_dict = {0: colors[0], 0.5: colors[1], 1: colors[2], 2: colors[3]}
    alpha_beta_no_tradeoff_config['color'] = alpha_beta_no_tradeoff_config['shutter_size'].apply(lambda x: color_dict[x])
    fig = plt.subplot(2,1,1)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='0',
                      markerfacecolor=color_dict[0], markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='0.5',
                      markerfacecolor=color_dict[0.5], markersize=10),
                  Line2D([0], [0], marker='o', color='w', label='1',
                      markerfacecolor=color_dict[1], markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='2',
                      markerfacecolor=color_dict[2], markersize=10)]
    fig.legend(labels=['shutter=0','shutter=0.5','shutter=1','shutter=2'], handles=legend_elements)

    plt.xlabel("Computation reduction")
    plt.ylabel("Instances solved [%]")


    # same analysis for configuration where there is a tradeoff between computation and correctness
    alpha_beta_tradeoff_config = alpha_beta_shutter_pareto.loc[(alpha_beta_shutter_pareto['max_moves']==30) & (alpha_beta_shutter_pareto['noise_sig']==0.5) & (alpha_beta_shutter_pareto['k']==5) & (alpha_beta_shutter_pareto['board']=='6_easy_full')]
    alpha_beta_tradeoff_config_all = alpha_beta_shutter.loc[(alpha_beta_shutter['max_moves']==30) & (alpha_beta_shutter['noise_sig']==0.5) & (alpha_beta_shutter['k']==5) & (alpha_beta_shutter['board']=='6_easy_full')]

    alpha_beta_tradeoff_config['inverse_computation'] = alpha_beta_tradeoff_config['numberOfHeuristicComp'].apply(lambda x: 1-(x/alpha_beta_tradeoff_config['numberOfHeuristicComp'].values.max()))
    alpha_beta_tradeoff_config_all['inverse_computation'] = alpha_beta_tradeoff_config_all['numberOfHeuristicComp'].apply(lambda x: 1.0-(float(x)/alpha_beta_tradeoff_config_all['numberOfHeuristicComp'].values.max()))

    alpha_beta_shutter_pareto_tradeoff = alpha_beta_tradeoff_config_all.groupby(['shutter_size']).agg(aggregations).reset_index()
    colors = sns.color_palette("hls", 4)
    color_dict = {0: colors[0], 0.5: colors[1], 1: colors[2], 2: colors[3]}
    alpha_beta_tradeoff_config['color'] = alpha_beta_tradeoff_config['shutter_size'].apply(lambda x: color_dict[x])

    x = alpha_beta_shutter_pareto_tradeoff[('inverse_computation','mean')]
    y = alpha_beta_shutter_pareto_tradeoff[('correct','mean')]
    x_err = alpha_beta_shutter_pareto_tradeoff[('inverse_computation','sem')]
    y_err = alpha_beta_shutter_pareto_tradeoff[('correct','sem')]

    fig = plt.subplot(2,1,2)
    fig.tick_params(axis='both', which='major', labelsize=10)
    for i in range(4):
        plt.errorbar(x[i], y[i], xerr=x_err[i], yerr=y_err[i], lw=2, capsize=5, capthick=2, color=colors[i],zorder=1)

    plt.xlabel("Computation reduction")
    plt.ylabel("Instances solved [%]")

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='0',
                      markerfacecolor=color_dict[0], markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='0.5',
                      markerfacecolor=color_dict[0.5], markersize=10),
                  Line2D([0], [0], marker='o', color='w', label='1',
                      markerfacecolor=color_dict[1], markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='2',
                      markerfacecolor=color_dict[2], markersize=10)]
    fig.legend(labels=['shutter=0','shutter=0.5','shutter=1','shutter=2'], handles=legend_elements)

    plt.tight_layout(pad=2.5)

    plt.show()


#### SI tables and figures ####

def heuristic_likelihoods_first_moves(heuristics_probs_filename):
    """ Table S2 (SI) """
    epsilon = 0.0001
    heuristic_probs_df = pd.read_csv(heuristics_probs_filename)
    heuristic_probs_df = heuristic_probs_df.loc[heuristic_probs_df.move_number_in_path == 1]
    heuristic_probs_df['log_likelihood'] = heuristic_probs_df.prob_move.apply(lambda x: math.log(max(x, epsilon)))
    heuristics = heuristic_probs_df.heuristic.unique()
    for h in heuristics:
        heuristic_probs_df_h = heuristic_probs_df.loc[heuristic_probs_df.heuristic == h]
        heuristic_probs_df_h = heuristic_probs_df_h[['heuristic', 'log_likelihood', 'prob_move']]
        print(h)
        # print(heuristic_probs_df_h.head())

        print(bootstrap_mean(heuristic_probs_df_h.log_likelihood.values))



def time_between_analysis(dynamics):
    """Figure S2A: time between moves analysis"""

    dynamics = dynamics[['userid','board_name','time_rel','prev_action','action','shutter','move_number_in_path','path_number','time_from_action','time_from_click','player']]
    userids = dynamics.userid.unique()
    print(dynamics.shape[0])
    dynamics = dynamics[(dynamics.move_number_in_path > 1) | (dynamics.path_number > 1)]
    dynamics = dynamics[(dynamics.time_from_click<=10)]

    dynamics = dynamics[(dynamics.action == 'click')]
    n = 0
    median_reg = []
    median_reset = []
    median_reset_undo = []
    correlations = []

    example_users = ['4cf589a5', '7dd704e0', '7df4e828' ,'cccfd3ac']

    bins = np.arange(0, 10, 0.5)
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    user_df = dynamics[dynamics['userid'] == example_users[0]]
    user_df = user_df[user_df.player == 1]
    user_reg_actions = user_df[user_df.prev_action == 'click']
    ax = sns.distplot(user_reg_actions['time_from_action'], bins=bins)
    ax.set(xlim=(0, 10))
    ax.set_xlabel('Time from previous move', fontsize=14)
    ax.set_ylabel('Percent of moves', fontsize=14)

    plt.subplot(1, 3, 2)
    user_df = dynamics[dynamics['userid'] == example_users[1]]
    user_df = user_df[user_df.player == 1]
    user_reg_actions = user_df[user_df.prev_action == 'click']
    ax = sns.distplot(user_reg_actions['time_from_action'], bins=bins)
    ax.set(xlim=(0, 10))
    ax.set_xlabel('Time from previous move', fontsize=14)
    ax.set_ylabel('Percent of moves', fontsize=14)

    plt.subplot(1, 3, 3)
    user_df = dynamics[dynamics['userid'] == example_users[3]]
    user_df = user_df[user_df.player == 1]
    user_reg_actions = user_df[user_df.prev_action == 'click']
    ax = sns.distplot(user_reg_actions['time_from_action'], bins=bins)
    ax.set(xlim=(0, 10))
    ax.set_xlabel('Time from previous move', fontsize=14)
    ax.set_ylabel('Percent of moves', fontsize=14)

    plt.tight_layout(pad=2.5)
    plt.show()


def auto_or_thinking(row):
    if row['move_number_in_path'] == 1:
        return 'thinking'
    elif row['time_from_click'] <= 2.5:
        return 'auto'
    else:
        return 'thinking'


def time_between_analysis_immediate_thinking(dynamics):
    """Figure S2B and S2C: time between moves analysis"""

    dynamics = dynamics[['userid','board_name','time_rel','prev_action','action','shutter','move_number_in_path','path_number','time_from_action','time_from_click','player','fitted_heuristic']]
    userids = dynamics.userid.unique()

    dynamics = dynamics[(dynamics.action == 'click')]

    # get users without undo actions
    uids = []
    for uid in userids:
        user_df = dynamics[dynamics['userid'] == uid]
        if user_df[user_df.prev_action == 'undo'].shape[0] == 0:
            uids.append(uid)

    # analyze times
    dynamics = dynamics[dynamics['userid'].isin(uids)]
    userids = dynamics.userid.unique()

    dynamics['act_type'] = dynamics.apply(auto_or_thinking, axis=1)

    dynamics = dynamics.loc[(dynamics['path_number'] > 1) | (dynamics['move_number_in_path'] > 1)]

    df = pd.read_csv('stats/thinking_auto_times_no_first_act.csv')
    print('median', df['auto_actions'].median)
    print('mean auto', bootstrap_mean(df['auto_actions'].values))

    # check for thinking-# following auto actions correlations
    thinking_times = []
    auto_actions_num = []
    curr_auto_counter = np.nan
    curr_thinking_time = np.nan

    for index, row in dynamics.iterrows():
        if row['act_type'] == 'thinking':
            if ((curr_auto_counter is not np.nan) & (curr_thinking_time is not np.nan)):
                thinking_times.append(curr_thinking_time)
                auto_actions_num.append(curr_auto_counter)
                curr_auto_counter = 0
                curr_thinking_time = row['time_from_action']
            else:
                curr_auto_counter = 0
                curr_thinking_time = row['time_from_action']
        else:
            curr_auto_counter += 1

    thinking_auto_df = pd.DataFrame({'thinking_time': thinking_times, 'auto_actions': auto_actions_num})
    print(spearmanr_ci(np.asarray(thinking_times), np.asarray(auto_actions_num)))

    thinking_auto_df['thinking_cat'] = pd.qcut(thinking_auto_df['thinking_time'], 3, labels=['short', 'medium', 'long'])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    ax = sns.regplot(x="thinking_time", y="auto_actions", data=thinking_auto_df, fit_reg=False)
    plt.xlim((0,20))
    plt.ylim((0,5.5))
    ax.set_xlabel('Thinking time (time until action)', fontsize=14)
    ax.set_ylabel('Number of fast moves \n following action', fontsize=14)

    users_thinking_time_auto_correlations = []
    # same analysis, per player
    for userid in userids:
        user_df = dynamics[dynamics['userid'] == userid]

        thinking_times = []
        auto_actions_num = []
        curr_auto_counter = np.nan
        curr_thinking_time = np.nan

        for index, row in user_df.iterrows():
            if row['act_type'] == 'thinking':
                if ((curr_auto_counter is not np.nan) & (curr_thinking_time is not np.nan)):
                    thinking_times.append(curr_thinking_time)
                    auto_actions_num.append(curr_auto_counter)
                    curr_auto_counter = 0
                    curr_thinking_time = row['time_from_action']
                else:
                    curr_auto_counter = 0
                    curr_thinking_time = row['time_from_action']
            else:
                curr_auto_counter += 1


        users_thinking_time_auto_correlations.append(stats.spearmanr(thinking_times, auto_actions_num)[0])

    users_thinking_time_auto_correlations = [x for x in users_thinking_time_auto_correlations if str(x) != 'nan']
    print('median: ',np.median(users_thinking_time_auto_correlations))
    print('mean: ', np.mean(users_thinking_time_auto_correlations))
    plt.subplot(1, 2, 2)
    ax = sns.distplot(users_thinking_time_auto_correlations, norm_hist=True)
    ax.set_xlabel('Correlation between thinking time \n and number of following fast moves', fontsize=14)
    ax.set_ylabel('Percent of participants', fontsize=14)
    plt.tight_layout(pad=2.5)
    plt.show()


def make_solution_time_actions_figure(dynamics):
    ''' figure S3A, S3B and S3C: solution times and number of actions figures'''

    aggregations = {
        # total time
        'time_rel': 'max',
        # total number of search moves
        'move_number': 'max'
    }

    # take only columns we need
    dynamics_min = dynamics[['userid', 'board_name',  'size_type', 'condition', 'time_rel', 'move_number', 'solved']]
    data = dynamics_min.groupby(['userid','size_type', 'condition', 'solved'], as_index=False).agg(aggregations)
    data['timeMinutes'] = data['time_rel'].apply(lambda x: x/60000.0)
    data['condition'] = data['condition'].apply(lambda x: 'truncated' if x=='pruned' else 'full')
    data['solutionValidationCorrect'] = data['solved'].map({'wrong': 0, 'solvedCorrect':0, 'validatedCorrect': 1})

    # rename boards according to paper naming
    name_mapping = {'6_easy': 'I', '6_hard': 'II', '10_easy': 'III', '10_hard': 'IV', '10_medium': 'V'}
    print(spearmanr_ci(data['timeMinutes'], data['move_number']))
    print('median: ', data['timeMinutes'].median())
    print('min: ', data['timeMinutes'].min())
    print('max: ', data['timeMinutes'].max())

    # filter data by correct/wrong participants
    data_correct = data[data['solutionValidationCorrect'] == 1]
    data_wrong = data[data['solutionValidationCorrect'] == 0]
    print('corr within correct',spearmanr_ci(data_correct['timeMinutes'], data_correct['move_number']))
    print('corr within wrong', spearmanr_ci(data_wrong['timeMinutes'], data_wrong['move_number']))
    print('median time correct: ', data_correct['timeMinutes'].median())
    print('median time wrong: ', data_wrong['timeMinutes'].median())
    print('mean time correct: ', data_correct['timeMinutes'].mean())
    print('mean time wrong: ', data_wrong['timeMinutes'].mean())

    print('median moves correct: ', data_correct['move_number'].median())
    print('median moves wrong: ', data_wrong['move_number'].median())
    print('mean moves correct: ', data_correct['move_number'].mean())
    print('mean moves wrong: ', data_wrong['move_number'].mean())

    for board in data['size_type'].unique():
        data_board  = data[data['size_type'] == board]
        data_correct = data_board[data_board['solutionValidationCorrect'] == 1]
        data_wrong = data_board[data_board['solutionValidationCorrect'] == 0]
        print(board)
        print('actions correct: ', data_correct['move_number'].median())
        print('actions wrong: ', data_wrong['move_number'].median())
        print('actions correct: ', data_correct['move_number'].mean())
        print('actions wrong: ', data_wrong['move_number'].mean())

    # figure S3C
    sns.set(font_scale=1.0, style="whitegrid")
    ax = sns.lmplot(x='timeMinutes', y='move_number', data=data, fit_reg=True, scatter_kws={"s": 10})
    ax = ax.set_axis_labels('Solution Time (min.)', 'Search Size')
    ax.set(ylim=(0, 100))
    ax.set(xlim=(0, 11))
    plt.show()


    # figures S3A S3B
    plt.subplot(1,2,1)
    data['board_name'] = data['size_type'].apply(lambda x: name_mapping[x])
    ax = sns.barplot(x="board_name", y="timeMinutes", hue="condition", data=data, ci=68, n_boot=1000, order=['I','II','III','IV','V'])
    ax.set_xlabel('Board', fontsize=14)
    ax.set_ylabel('Solution Time (min.)', fontsize=14)
    plt.legend(loc='best', fontsize=9)

    plt.subplot(1,2,2)
    ax = sns.barplot(x="board_name", y="move_number", hue="condition", data=data, ci=68, n_boot=1000, order=['I','II','III','IV','V'])
    ax.set_xlabel('Board', fontsize=14)
    ax.set_ylabel('Search Size', fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout(pad=2)
    plt.show()


def search_time_correctness_figure(alpha_beta_df, behavioral_df):
    '''
    Figure S3D: board complexity and search time
    '''

    # generate behavioral data with action counts
    behavioral_clicks_df = behavioral_df.loc[((behavioral_df['action']=='click'))]
    behavioral_clicks_df['time_seconds'] = behavioral_clicks_df['time_rel'].apply(lambda x: x/60000.0)
    solution_times_df = behavioral_clicks_df.groupby(['userid','solved','board_name', 'board_size','board_type','condition'], as_index=False)['time_seconds'].max()


    # correlation alpha-beta interaction and participants moves
    solution_times_board_means_df = solution_times_df.groupby(['board_name', 'board_size','board_type','condition'], as_index=False)['time_seconds'].mean()
    solution_times_board_means_df['type'] = solution_times_board_means_df['board_type'].map({'hard': 'HC', 'easy': 'MC', 'medium': 'DC'})
    solution_times_board_means_df['size'] = solution_times_board_means_df['board_size']
    solution_times_board_means_df['condition'] = solution_times_board_means_df['condition'].map({'full': 'full', 'pruned': 'truncated'})

    alpha_beta_df['heuristic_name'] = alpha_beta_df['heuristic_name'].map({'density':'density', 'linear':  'linear','non-linear':'non-linear', 'non-linear-interaction': 'interaction','blocking':'blocking', 'participants':'participants'})
    alphaBetaInteraction = alpha_beta_df.loc[alpha_beta_df['heuristic_name'] == 'interaction']


    behavioral_agg_df = behavioral_clicks_df.groupby(['userid','board_size','board_type','condition','board_name','solved']).time_seconds.max().reset_index()
    behavioral_agg_df['solutionAndValidationCorrectPercent'] = behavioral_agg_df['solved'].map({'wrong': 0, 'solvedCorrect':0, 'validatedCorrect': 100})
    behavioral_agg_df['solutionAndValidationCorrect'] = behavioral_agg_df['solved'].map({'wrong': 0, 'solvedCorrect':0, 'validatedCorrect': 1})
    behavioral_agg_df['solutionCorrect'] = behavioral_agg_df['solved'].map({'wrong': 0, 'solvedCorrect':1, 'validatedCorrect': 1})
    behavioral_agg_df['type'] = behavioral_agg_df['board_type'].map({'hard': 'HC', 'easy': 'MC', 'medium': 'DC'})
    behavioral_agg_df['size'] = behavioral_agg_df['board_size']
    behavioral_agg_df['condition'] = behavioral_agg_df['condition'].map({'full': 'full', 'pruned': 'truncated'})


    for board_name in BOARD_NAMES:
        action_counts_board = solution_times_df.loc[solution_times_df['board_name'] == board_name]
        print(board_name)
        print('median', action_counts_board['time_seconds'].median())
        print('mean', action_counts_board['time_seconds'].mean())

    behavioral_agg_df['action'] =  behavioral_agg_df['time_seconds']
    alpha_beta_behavioral_df = pd.merge(alphaBetaInteraction, behavioral_agg_df, on=['size','type','condition'], how='left')

    alpha_beta_behavioral__interaction_df = alpha_beta_behavioral_df.loc[alpha_beta_behavioral_df['heuristic_name'] == 'interaction']

    print('Spearman correlations:')
    print('Correlation complexity-participants number of moves', spearmanr_ci(alpha_beta_behavioral__interaction_df.action.values, alpha_beta_behavioral__interaction_df.moves.values))
    print('Correlation complexity-participants correctness', spearmanr_ci(alpha_beta_behavioral__interaction_df.solutionAndValidationCorrect.values, alpha_beta_behavioral__interaction_df.moves.values))

    # plot
    ax = sns.regplot(x="moves", y="time_seconds", data=alpha_beta_behavioral_df,  x_estimator=np.mean, ci=68, color="b", fit_reg=False)
    sns.regplot(x="moves", y="moves", data=alpha_beta_behavioral_df,  x_estimator=np.mean, ci=68, color="silver", fit_reg=False, ax=ax)
    ax.set(xscale="log")
    ax.set(yscale="log")
    ax.set(xlim=(10, 200000))
    ax.set_xlabel('Board Complexity', fontsize=14)
    ax.set_ylabel('Search Time', fontsize=14)
    ax.tick_params(labelsize=12)


    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Participants Search Time',
                          markerfacecolor='b', markersize=9),
                       Line2D([0], [0], marker='o', color='w', label='Alpha-Beta Search Size',
                          markerfacecolor='silver', markersize=9)]
    ax.legend(handles=legend_elements, fontsize=11)

    ax.set(xlabel='Board Complexity', ylabel='Search Time (Min.)')
    plt.tight_layout(pad=1)
    plt.show()


def alpha_beta_vs_participants_number_of_moves(alpha_beta_participants_data):
    """Figure S4: The number of moves explored by the alpha-beta algorithm using the different scoring strategies, and the number of moves tried by participants"""
    alpha_beta_participants_data['heuristic_name'] = alpha_beta_participants_data['heuristic_name'].map({'density':'Density', 'linear':  'Linear','non-linear':'Non-linear', 'non-linear-interaction': 'Interaction','blocking':'Forcing', 'participants':'Participants'})
    sns.set(font_scale=1.2, style='whitegrid')

    alpha_beta_participants_data['board'] = alpha_beta_participants_data['board'].map({'6 MC full': 'I full', '6 MC truncated': 'I truncated','10 MC full': 'III full','10 MC truncated':'III truncated','6 HC full': 'II full', '6 HC truncated':'II truncated','10 HC full':'IV full','10 HC truncated':'IV truncated', '10 DC full': 'V full','10 DC truncated':'V truncated'})
    ax = sns.factorplot(x="board", y="moves",  scale= 0.5, data=alpha_beta_participants_data, hue="heuristic_name", n_boot=1000, order=['I truncated', 'I full', 'II truncated', 'III truncated', 'V truncated','II full', 'IV truncated', 'III full',   'IV full', 'V full'],  markers=["<","1","2","3","4","*"],linestyles=["-","-","-","-","-", "--"], legend_out=False, legend=False)
    ax.fig.get_axes()[0].set_yscale('log')
    plt.ylim(0, 200000)

    ax.set(xlabel='Board', ylabel='Number of Moves')
    lw = ax.ax.lines[0].get_linewidth()
    plt.setp(ax.ax.lines,linewidth=lw)
    plt.legend(loc='best')
    plt.show()
    exit()


def make_participant_heuristic_fit_figure(heuristic_fit_df):
    """Figure S5: scoring strategy fit to participants"""
    heuristic_fit_counts = heuristic_fit_df.groupby(['fitted_heuristic'], as_index=False)['userid'].count()
    total_participants = heuristic_fit_counts.userid.sum()
    heuristic_fit_counts['percent_participants'] = heuristic_fit_counts['userid'].apply(lambda x: x/total_participants)

    print(heuristic_fit_counts.head())
    print(heuristic_fit_counts['percent_participants'].sum())

    heuristics = heuristic_fit_counts.fitted_heuristic.unique()


    # can uncomment if wish to re-do bootstrapping
    heuristic_list = []
    percent_list = []
    # for i in range(1000):
    #     bootstrap_sample = heuristic_fit_df.sample(frac=1.0, replace=True)
    #     heuristic_fit_counts = bootstrap_sample.groupby(['fitted_heuristic'], as_index=False)['userid'].count()
    #     total_participants = heuristic_fit_counts.userid.sum()
    #     heuristic_fit_counts['percent_participants'] = heuristic_fit_counts['userid'].apply(lambda x: x/total_participants)
    #     for heuristic in heuristics:
    #         heuristic_list.append(heuristic)
    #         percent_list.append(heuristic_fit_counts[heuristic_fit_counts['fitted_heuristic']==heuristic]['percent_participants'].values[0])
    #
    # bootstrap_df = pd.DataFrame({'heuristic': heuristic_list, 'percent_participants': percent_list})
    # bootstrap_df.to_csv('stats/test_bootstrap_heuristic_fit.csv')

    # read pre-processed data
    bootstrap_df = pd.read_csv('stats/test_bootstrap_heuristic_fit.csv')
    bootstrap_df['heuristic'] = bootstrap_df['heuristic'].map({'density':'Density', 'linear':  'Linear','non-linear':'Non-linear', 'interaction': 'Interaction','blocking':'Forcing'})

    heuristics_order = ['Density','Linear','Non-linear','Interaction', 'Forcing']
    ax = sns.barplot(x='heuristic', y='percent_participants', ci='boot_custom', data=bootstrap_df, order=heuristics_order)
    ax.set_xlabel('Scoring strategy', fontsize=14)
    ax.set_ylabel('Percent participants', fontsize=14)
    plt.show()

    print('----stats----')
    for h in heuristics_order:
        print(h)
        data_h = bootstrap_df.loc[bootstrap_df.heuristic == h]
        print(np.round(np.mean(data_h.percent_participants.values),2))
        print(np.round(np.percentile(data_h.percent_participants.values, q=(100*0.05/2, 100*(1-0.05/2))),2))


def heuristic_fit_sensitivity_analysis(heuristic_sensitivity_win_score_filename, heuristic_sensitivity_block_score_filename):
    """Figure S7: heuristic sensitivity analysis """
    epsilon = 0.0001

    heuristics_sensitivity_win_score_df = pd.read_csv(heuristic_sensitivity_win_score_filename)
    heuristics_sensitivity_win_score_df = heuristics_sensitivity_win_score_df.loc[heuristics_sensitivity_win_score_df.action == 'click']
    heuristics_sensitivity_win_score_df = heuristics_sensitivity_win_score_df[['userid', 'heuristic', 'win_score','prob_move']]
    heuristics_sensitivity_win_score_df['log_likelihood'] = heuristics_sensitivity_win_score_df['prob_move'].apply(lambda x: math.log(max(x, epsilon)))
    heuristics_sensitivity_scores =  heuristics_sensitivity_win_score_df.groupby(['userid','win_score','heuristic'], as_index=False).log_likelihood.mean()

    win_scores = heuristics_sensitivity_scores.win_score.unique()
    heuristics = heuristics_sensitivity_scores.heuristic.unique()
    print(heuristics)
    changes = []
    folds = []
    heuristics_names = []
    heuristics_sensitivity_score_25 = heuristics_sensitivity_scores.loc[heuristics_sensitivity_scores['win_score'] == 25]
    for win_score in win_scores:
        heuristics_sensitivity_score = heuristics_sensitivity_scores.loc[heuristics_sensitivity_scores['win_score'] == win_score]
        fold = win_score/25.0
        for user in heuristics_sensitivity_score['userid'].unique():
            user_scores_25 = heuristics_sensitivity_score_25.loc[heuristics_sensitivity_score_25['userid'] == user]
            user_scores = heuristics_sensitivity_score.loc[heuristics_sensitivity_score['userid'] == user]
            for h in heuristics:
                change = abs((user_scores[user_scores.heuristic == h].log_likelihood.values[0]/user_scores_25[user_scores_25.heuristic == h].log_likelihood.values[0]))
                if change < 1:
                    change = 1.0/change
                change = change - 1
                changes.append(change)
                folds.append(int(fold))
                heuristics_names.append(h)

    changes_dict = {'fold': folds, 'changes':changes, 'heuristic': heuristics_names}
    changes_df = pd.DataFrame(changes_dict)

    for h in heuristics:
        print(h)
        max_fold_df = changes_df.loc[(changes_df['fold'] == 100) & (changes_df['heuristic'] == h) ]
        print(bootstrap_mean(max_fold_df['changes'].values))

    heuristics_sensitivity_blocking_df = pd.read_csv(heuristic_sensitivity_block_score_filename)
    heuristics_sensitivity_blocking_df = heuristics_sensitivity_blocking_df[['userid', 'heuristic', 'blocking_score','prob_move']]
    heuristics_sensitivity_blocking_df['log_likelihood'] = heuristics_sensitivity_blocking_df['prob_move'].apply(lambda x: math.log(max(x, epsilon)))
    heuristics_sensitivity_blocking_scores =  heuristics_sensitivity_blocking_df.groupby(['userid','blocking_score','heuristic'], as_index=False).log_likelihood.mean()


    blocking_scores = heuristics_sensitivity_blocking_scores.blocking_score.unique()
    heuristics = heuristics_sensitivity_blocking_scores.heuristic.unique()
    print(heuristics)
    changes = []
    folds = []
    heuristics_names = []
    heuristics_sensitivity_score_2 = heuristics_sensitivity_blocking_scores.loc[heuristics_sensitivity_blocking_scores['blocking_score'] == 2]
    for block_score in blocking_scores:
        heuristics_sensitivity_score = heuristics_sensitivity_blocking_scores.loc[heuristics_sensitivity_blocking_scores['blocking_score'] == block_score]
        fold = block_score/2.0
        for user in heuristics_sensitivity_score['userid'].unique():
            user_scores_2 = heuristics_sensitivity_score_2.loc[heuristics_sensitivity_score_2['userid'] == user]
            user_scores = heuristics_sensitivity_score.loc[heuristics_sensitivity_score['userid'] == user]
            for h in heuristics:
                change = abs((user_scores[user_scores.heuristic == h].log_likelihood.values[0]/user_scores_2[user_scores_2.heuristic == h].log_likelihood.values[0]))
                if change < 1:
                    change = 1.0/change
                change = change - 1
                changes.append(change)
                folds.append(int(fold))
                heuristics_names.append(h)

    blocking_changes_dict = {'fold': folds, 'changes':changes, 'heuristic': heuristics_names}
    blocking_changes_df = pd.DataFrame(blocking_changes_dict)

    for h in heuristics:
        print(h)
        max_fold_df = blocking_changes_df.loc[(blocking_changes_df['fold'] == 100) & (blocking_changes_df['heuristic'] == h) ]
        print(bootstrap_mean(max_fold_df['changes'].values))

    plt.subplot(1,2,1)
    ax = sns.pointplot('fold', 'changes', hue='heuristic', data=changes_df, legend_out=False, legend=False)
    plt.gca().legend().set_title('')
    plt.legend(loc='best')
    ax.set_ylabel('Mean percent change in log-likelihood', fontsize=16)
    ax.set_xlabel('Fold change in winning score', fontsize=16)
    ax.tick_params(labelsize=14)

    plt.subplot(1,2,2)
    ax = sns.pointplot('fold', 'changes', data=blocking_changes_df)
    ax.set_ylabel('Mean percent change in log-likelihood', fontsize=16)
    ax.set_xlabel('Fold change in blocking score', fontsize=16)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.show()


def generate_predicted_vs_observed_path_correlations():
    """Figure S8: correlations between path probabilities in behavioral data and predicted probabilities from scoring strategies, by path length"""
    correlations_df = pd.read_csv('stats/heuristics_participants_paths_correlations_lengths_sep_spearman_10_0.01.csv')
    correlations_df['heuristic'] = correlations_df['heuristic'].map({'density':'Density', 'linear':  'Linear','non-linear':'Non-linear', 'interaction': 'Interaction','blocking':'Forcing'})
    correlations_df.dropna(inplace=True)
    heuristics = ['Density', 'Linear', 'Non-linear', 'Interaction', 'Forcing']

    for heuristic in heuristics:
        correlations_heuristic = correlations_df.loc[correlations_df['heuristic'] == heuristic]
        print(heuristic)
        print('spearman r')
        print(spearmanr_ci(correlations_heuristic['path_length'].values, correlations_heuristic['correlation'].values))
    ax = sns.pointplot('path_length', 'correlation', hue='heuristic',  data=correlations_df, legend_out=False, legend=False)
    plt.gca().legend().set_title('')


    ax.set_ylabel('Correlation', fontsize=16)
    ax.set_xlabel('Path length', fontsize=16)
    ax.tick_params(labelsize=14)

    plt.legend(loc='best')
    plt.show()


def make_mcts_entropy_full_pruned_fig(num_simulations=500):
    """Figure S11: MCTS entropies full vs. truncated """

    entropies_data = pd.read_csv('stats/entropies_data_5000nodesk5_simulation.csv')  # entropies data from MCTS runs

    ax = sns.barplot(x='condition', y='entropy',  n_boot=1000, ci=68, data=entropies_data)
    ax.set_xlabel('Condition', fontsize=14)
    ax.set_ylabel('Entropy', fontsize=14)
    ax.tick_params(labelsize=12)
    plt.title('MCTS 5000 nodes k=5 entropies')
    plt.show()

    entropies_full_df = entropies_data.loc[entropies_data.condition == 'full']
    entropies_truncated_df = entropies_data.loc[entropies_data.condition == 'truncated']
    full_ci = np.nanpercentile(entropies_full_df.entropy.values, q=(100*0.05/2, 100*(1-0.05/2)))
    pruned_ci = np.nanpercentile(entropies_truncated_df.entropy.values, q=(100*0.05/2, 100*(1-0.05/2)))

    print('MCTS 5000 nodes k=5 mean entropies:')
    print ('Mean entropy full boards', entropies_full_df.entropy.mean())
    print(full_ci)
    print ('Mean entropy truncated boards', entropies_truncated_df.entropy.mean())
    print(pruned_ci)
    print('Mann whitney test for differences in entropies between full and truncated boards:')
    print(stats.mannwhitneyu(entropies_full_df.entropy.values, entropies_truncated_df.entropy.values))
    print(rank_biserial_effect_size(entropies_full_df.entropy.values, entropies_truncated_df.entropy.values))



if __name__== "__main__":
    # Uncomment the lines to run each figure seperately

    sns.set(font_scale=1., style="whitegrid")

    # main data file - more than 3 moves participants:
    data = pd.read_csv("stats/dynamics_shutter_blindness_heuristic_filtered.csv")

    # generating figure 2 - search size and correctness
    # alphaBetaFull = pd.read_csv("stats/alphaBetaSearchData.csv")  # results of alpha-beta runs
    # search_size_correctness_figure(alphaBetaFull, data)


    # generating figure 3a - heatmap comparing participants' first moves with the interaction heuristic
    # generate_participants_heuristics_heatmpaps('stats/avg_people_clicks_firstPaths.json', 'interaction')


    # generating figure 3b
    # make_emd_correlations_fig()

    # generating figures 3C and 3D
    # generate_correlation_figures()


    # generating figures 4A and 4B
    # generate_entropy_heatmap(data, '10_easy')

    # generating figure 4C
    # plot_entropies_full_truncated('stats/entropies_all_first.csv')

    # generating figure 5A
    # shutter_blindness_df = pd.read_csv('stats/shutter_blindness_avg_open.csv')
    # shutter_correct_df = pd.read_csv('stats/shutter_correctness_avg_board_open.csv')
    # make_shutter_blindness_figure(shutter_correct_df,shutter_blindness_df)

    # generating figure 5B and figure S13
    # comp_shutter_all_50moves = pd.read_csv('stats/ab_config_shutterab_pareto_02022020_k3_k5_cats_50moves.csv')
    # comp_shutter_blindness(comp_shutter_all_50moves)


    # generating figure 6A
    # ab_pareto_tradeoff_example_figs()

    # generating figure 6B
    # ab_pareto_tradeoff_analysis()

    # generating data for table S2
    # heuristic_likelihoods_first_moves('stats/heuristics_byMove_player_withPaths.csv')

    # generating figure S2A
    # time_between_analysis(data)

    # generating figures S2B and S2C
    # time_between_analysis_immediate_thinking(data)

    # generating figure S3A, S3B and S3C
    # make_solution_time_actions_figure(data)

    # generating figure S3D
    # alphaBetaFull = pd.read_csv("stats/alphaBetaSearchData.csv")  # results of alpha-beta runs
    # search_time_correctness_figure(alphaBetaFull, data)

    # generating figure S4
    # alphaBetaFull = pd.read_csv("stats/alphaBetaSearchData.csv")  # results of alpha-beta runs
    # alpha_beta_vs_participants_number_of_moves(alphaBetaFull)

    # generating figure S5
    # make_emd_correlations_fig(with_mcts=True)

    # generating figure S6
    # heuristics_fitted_df = pd.read_csv('stats/participants_heuristics_fit.csv')
    # make_participant_heuristic_fit_figure(heuristics_fitted_df)

    # generating figure S7
    # heuristic_fit_sensitivity_analysis('stats/heuristics_sensitivity.csv', 'stats/heuristics_byMove_player_sensitivity_blockingVals.csv')

    # generating figure S8
    # generate_predicted_vs_observed_path_correlations()

    # generating figures S9 and S10
    # plot_entropies_full_truncated('stats/entropies_raw_all_all.csv',
    #                               solvers_filename='stats/entropies_raw_solvers_all.csv',
    #                               nonsolvers_filename='stats/entropies_raw_non-solvers_all.csv')

    # generating figure S11
    # make_mcts_entropy_full_pruned_fig()