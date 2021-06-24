"""
Main script used to make statistics of models against other players.

A very unfortunate deficiency: players.py was not used here.
Instead, a "no_playouts" parameter was passed to MCTS.

"""

from __future__ import print_function

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re
import ast
import io
import sys
import PIL


# from AlphaZeroAnalyses.aux_for_models_statistics import *

# warnings.simplefilter("error", np.VisibleDeprecationWarning)
MAX_POOL = 28
MAX_POOL = 28



########################################################
########################################################
########    Plots that we experimented with     ########
########################################################
########################################################




########################################################
########################################################
#######    Displayed plots and summery making    #######
########################################################
########################################################


def produce_model_vs_opponent_summery_excel(model_name, board_size, opponent_name, res_path, game_statistics_path):
    limits_shutter = [None, 0] if board_size == 6 else [None, 0, 1, 2]

    if board_size == 6:
        all_boards_names = ["board 1 full", "board 1 truncated", "board 2 full", "board 2 truncated", "empty board"]
    else:
        all_boards_names = ["board 3 full", "board 3 truncated", "board 4 full", "board 4 truncated", "board 5 full",
                            "board 5 truncated", "empty board"]

    dfs = []
    indexes = []
    for index, shutter_limit in enumerate(limits_shutter):

        for board_name in all_boards_names:
            path = f"{game_statistics_path}{board_size}X{board_size}_statistics_limit_all_to_shutter_{shutter_limit}/vs {opponent_name}/{board_name}/all models {num_games} games results.xlsx"
            data = pd.read_excel(path, index_col=0)
            data = data.loc[model_name]
            indexes.append(f"{board_name}_{shutter_limit}")
            dfs.append(data)

    res = pd.DataFrame(dfs, index=indexes)

    os.makedirs(f"{res_path}plots_and_summery/statistics_summaries/", exist_ok=True)
    res.to_csv(f'{res_path}plots_and_summery/statistics_summaries/{opponent_name} vs {model_name}.csv')


def make_paper_plots_all_models(model_name_6, model_name_10, opponent_name, num_games,
                                fig_width, height,
                                game_statistics_path, limits_shutter=[None, 0], add_truncated=True, add_empty=False):


    mpl.rcParams.update({'font.size': 25})


    bars_dict = {index: [] for index in range(len(limits_shutter))}
    CI_dict = {index: [] for index in range(len(limits_shutter))}


    if add_truncated:
        all_boards_names_6 = ["board 1 full",
                              "board 1 truncated",
                              "board 2 full",
                              "board 2 truncated"
                              ]

        all_boards_names_10 = [
            "board 3 full",
            "board 3 truncated",
            "board 4 full",
            "board 4 truncated",
            "board 5 full",
            "board 5 truncated",
        ]

        all_boards_names_legend = ["I full",
                                   "I truncated",
                                   "II full",
                                   "II truncated",
                                   "III full",
                                   "III truncated",
                                   "IV full",
                                   "IV truncated",
                                   "V full",
                                   "V truncated",
                                   ]

    else:

        all_boards_names_6 = ["board 1 full",
                              "board 2 full",
                              ]

        all_boards_names_10 = [
            "board 3 full",
            "board 4 full",
            "board 5 full",
        ]

        all_boards_names_legend = ["I full",
                                   "II full",
                                   "III full",
                                   "IV full",
                                   "V full",
                                   ]


    for index, shutter_limit in enumerate(limits_shutter):

        for board_name in all_boards_names_6:
            path = f"{game_statistics_path}6X6_statistics_limit_all_to_shutter_{shutter_limit}/vs {opponent_name}/{board_name}/all models {num_games} games results.xlsx"

            data = pd.read_excel(path, index_col=0)

            bars_dict[index].append(100 * data.at[model_name_6, "no. wins"] / num_games)

            CI_dict[index].append(data.at[model_name_6, "CI_wins_losses"])



        for board_name in all_boards_names_10:
            path = f"{game_statistics_path}/10X10_statistics_limit_all_to_shutter_{shutter_limit}/vs {opponent_name}/{board_name}/all models {num_games} games results.xlsx"

            data = pd.read_excel(path, index_col=0)

            bars_dict[index].append(100 * data.at[model_name_10, "no. wins"] / num_games)

            CI_dict[index].append(data.at[model_name_10, "CI_wins_losses"])


        if add_empty:
            path = f"{game_statistics_path}6X6_statistics_limit_all_to_shutter_{shutter_limit}/vs {opponent_name}/empty board/all models {num_games} games results.xlsx"

            data = pd.read_excel(path, index_col=0)

            bars_dict[index].append(100 * data.at[model_name_6, "no. wins"] / num_games)
            CI_dict[index].append(data.at[model_name_6, "CI_wins_losses"])

            path = f"{game_statistics_path}10X10_statistics_limit_all_to_shutter_{shutter_limit}/vs {opponent_name}/empty board/all models {num_games} games results.xlsx"

            data = pd.read_excel(path, index_col=0)

            bars_dict[index].append(100 * data.at[model_name_10, "no. wins"] / num_games)
            CI_dict[index].append(data.at[model_name_10, "CI_wins_losses"])


    if len(bars_dict[0]) == 2:
        fig_width = fig_width / 3


    if len(all_boards_names_legend) == 10:
        fig_width = fig_width * 2


    fig, ax = plt.subplots(constrained_layout=True)

    fig.set_size_inches(fig_width, height)

    ind = np.arange(len(bars_dict[0]))
    width = 0.27

    alpha = 1

    ms = 40 if len(all_boards_names_legend) == 5 else 20

    ax.bar(ind - width / 1.5, bars_dict[0], width=width, color='#5f9e6e', alpha=alpha, label=f'No shutter limitation')
    ax.bar(ind + width / 1.5, bars_dict[1], width=width, color='#5874a2', alpha=alpha, label=f'Shutter = {limits_shutter[1]}')


    for index, ci in zip(ind, CI_dict[0]):

        ci = npstr2tuple(ci)
        ci_percent = (ci[0 ] *100, ci[1 ] *100)
        ax.plot((index - width / 1.5, index - width / 1.5), ci_percent, 'r_-', color='black', linewidth=4, mew=4, ms=ms)

    for index, ci in zip(ind, CI_dict[1]):
        ci = npstr2tuple(ci)
        ci_percent = (ci[0] * 100, ci[1] * 100)
        ax.plot((index + width / 1.5, index + width / 1.5), ci_percent, 'r_-', color='black', linewidth=4, mew=4, ms=ms)



    ax.legend(fancybox=False, shadow=False, fontsize=30, ncol=1)

    ax.set_xticks(ind)
    ax.set_xticklabels(all_boards_names_legend, fontsize=30, weight='bold')
    ax.set_ylabel("Game wins percentage ", fontsize=30, weight='bold')
    plt.locator_params(axis='y', nbins=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)


    limit_shutter_str = '_'.join([str(lim) for lim in limits_shutter])
    path = f"{game_statistics_path}plots_and_summery/{limit_shutter_str}_shutters_comparison_plots_for_all_models/"

    if not os.path.exists(path):
        os.makedirs(path)


    if add_truncated:
        plt.savefig(f"{path}{model_name_6} and {model_name_10} vs {opponent_name}.png", bbox_inches='tight')

    else:
        plt.savefig(f"{path}{model_name_6} and {model_name_10} vs {opponent_name} no truncated.png", bbox_inches='tight')

    plt.show()
    plt.close('all')


def npstr2tuple(s):
    # Remove space after [
    s = re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s = re.sub('[,\s]+', ', ', s)
    return tuple(np.array(ast.literal_eval(s)))


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # mpl.use('agg')
    # set_start_method("spawn")
    num_games = 1000
    #
    #
    # if len(sys.argv) == 1:
    game_statistics_path = '../stats/'
    # else:
    #     game_statistics_path = sys.argv[1]


    produce_model_vs_opponent_summery_excel("v9_1500", 6, "pure MCTS 500", game_statistics_path, game_statistics_path)
    produce_model_vs_opponent_summery_excel("v9_1500", 6, "pure MCTS 1000", game_statistics_path, game_statistics_path)

    produce_model_vs_opponent_summery_excel("v_01_1500", 10, "pure MCTS 500", game_statistics_path, game_statistics_path)
    produce_model_vs_opponent_summery_excel("v_01_1500", 10, "pure MCTS 1000", game_statistics_path, game_statistics_path)


    model_name_6 = "v9_1500"
    model_name_10 = "v_01_1500"
    fig_width = 15
    height = 10
    opponent_name = "pure MCTS 500"
    make_paper_plots_all_models(model_name_6, model_name_10, opponent_name, num_games, fig_width, height, game_statistics_path)



    model_name_6 = "v9_1500"
    model_name_10 = "v_01_1500"
    fig_width = 15
    height = 10
    opponent_name = "pure MCTS 1000"
    make_paper_plots_all_models(model_name_6, model_name_10, opponent_name, num_games, fig_width, height, game_statistics_path)

