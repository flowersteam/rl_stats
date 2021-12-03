"""
In this script we show how to test and plot RL results.

"""
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
font = {'family': 'normal',
        'size': 70}
matplotlib.rc('font', **font)
sys.path.append('../')
from rl_stats.tests import run_test, compute_central_tendency_and_error



def run_test_and_plot(data1,  # array of performance of dimension (n_steps, n_seeds) for alg 1
                      data2,  # array of performance of dimension (n_steps, n_seeds) for alg 2
                      point_every=1,  # evaluation frequency, one datapoint every X steps/episodes
                      test_id="welch",  # choose among ['t_test', "welch", 'mann_whitney', 'ranked_t_test', 'bootstrap', 'permutation']. Welch recommended,
                      # see paper.
                      confidence_level=0.01,  # confidence level alpha of the test
                      id_central='median',  # id of the central tendency ('mean' or 'median')
                      id_error=80,  # id of the error areas ('std', 'sem', or percentiles in ]0, 100]
                      legends=None,  # labels of the two input vectors
                      xlabel='training steps',  # label of the x axis
                      save=True,  # save in ./plot.png if True
                      downsampling_fact=5  # factor of downsampling on the x-axis for visualization purpose (increase for smoother plots)
                      ):

    # if they are paths, load the files as numpy arrays.
    if isinstance(data1, str):
        data1 = np.loadtxt(data1)
        data2 = np.loadtxt(data2)

    assert data1.ndim == 2, "data should be an array of dimension 2 (n_steps, n_seeds)"
    assert data2.ndim == 2, "data should be an array of dimension 2 (n_steps, n_seeds)"
    nb_steps = max(data1.shape[0], data2.shape[0])
    steps = [0]
    while len(steps) < nb_steps:
        steps.append(steps[-1] + point_every)
    steps = np.array(steps)
    if steps is not None:
        assert steps.size == nb_steps, "x should be of the size of the longest data array"

    sample_size1 = data1.shape[1]
    sample_size2 = data2.shape[1]

    # downsample for visualization purpose
    sub_steps = np.arange(0, nb_steps, downsampling_fact)
    steps = steps[sub_steps]
    nb_steps = steps.size

    # handle arrays of different lengths by padding with nans
    sample1 = np.zeros([nb_steps, sample_size1])
    sample1.fill(np.nan)
    sample2 = np.zeros([nb_steps, sample_size2])
    sample2.fill(np.nan)
    sub_steps1 = sub_steps[:data1.shape[0] // downsampling_fact]
    sub_steps2 = sub_steps[:data2.shape[0] // downsampling_fact]
    sample1[:data1[sub_steps1, :].shape[0], :] = data1[sub_steps1, :]
    sample2[:data2[sub_steps2, :].shape[0], :] = data2[sub_steps2, :]

    # test
    sign_diff = np.zeros([len(steps)])
    for i in range(len(steps)):
        sign_diff[i] = run_test(test_id, sample1[i, :], sample2[i, :], alpha=confidence_level)

    central1, low1, high1 = compute_central_tendency_and_error(id_central, id_error, sample1)
    central2, low2, high2 = compute_central_tendency_and_error(id_central, id_error, sample2)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    lab1 = plt.xlabel(xlabel)
    lab2 = plt.ylabel('performance')

    plt.plot(steps, central1, linewidth=10)
    plt.plot(steps, central2, linewidth=10)
    plt.fill_between(steps, low1, high1, alpha=0.3)
    plt.fill_between(steps, low2, high2, alpha=0.3)
    if legends is None:
        legends = ('alg 1', 'alg 2')
    leg = ax.legend(legends, frameon=False)

    # plot significative difference as dots
    idx = np.argwhere(sign_diff == 1)
    y = max(np.nanmax(high1), np.nanmax(high2))
    plt.scatter(steps[idx], y * 1.05 * np.ones([idx.size]), s=100, c='k', marker='o')

    # style
    for line in leg.get_lines():
        line.set_linewidth(10.0)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)

    if save:
        plt.savefig('./plot.png', bbox_extra_artists=(leg, lab1, lab2), bbox_inches='tight', dpi=100)

    plt.show()

if __name__ == '__main__':
    import argparse
    data1 = np.loadtxt('./data/sac_hc_all_perfs.txt')
    data2 = np.loadtxt('./data/td3_hc_all_perfs.txt')
    sample_size = 20
    data1 = data1[:300, np.random.randint(0, data1.shape[1], sample_size)]
    data2 = data2[:, np.random.randint(0, data1.shape[1], sample_size)]
    legends = ['SAC', 'TD3']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--data1', type=str, default=data1, help='path to text file containing array of performance of dimension (n_steps, n_seeds) for alg 1')
    add('--data2', type=str, default=data2, help='path to text file containing array of performance of dimension (n_steps, n_seeds) for alg 2')
    add('--point_every', type=int, default=1, help='evaluation frequency, one datapoint every X steps/episodes')
    add('--test_id', type=str, default="welch", help="choose in [t_test, welch, mann_whitney, ranked_t_test, bootstrap, permutation], welch recommended (see paper)")
    add('--confidence_level', type=float, default=0.01, help='confidence level alpha of the test')
    add('--id_central', type=str, default='median', help="id of the central tendency ('mean' or 'median')")
    add('--id_error', default=80, help="id of the error areas ('std', 'sem', or percentiles in ]0, 100]")
    add('--legends', default=None, help='labels of the two input vectors (str, str)')
    add('--xlabel', type=str, default='training episodes', help='label of the x axis, usually episodes or steps')
    add('--save', type=bool, default=True, help='save in ./plot.png if True')
    add('--downsampling_fact', type=int, default=5, help='factor of downsampling on the x-axis for visualization purpose (increase for smoother plots)')
    kwargs = vars(parser.parse_args())
    run_test_and_plot(**kwargs)