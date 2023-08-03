import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

sns.set_style('white', {'xtick.bottom': True, 'ytick.left': True})
# sns.set_context('notebook', font_scale=2)


def get_acts(result_path):
    model_data = pickle.load(open(result_path, 'rb'))
    input_act = model_data['input']
    image = model_data['image']
    # find preferred image
    img_idx = [np.argwhere(image == i) for i in range(8)]
    input_act_image = [
        input_act[i[:, 0], i[:, 1]].mean(axis=0) for i in img_idx]
    pref_image = np.argmax(np.stack(input_act_image), axis=0)

    return model_data, pref_image


def get_go_trials(model_data, pref_image):
    input_act = model_data['input']
    labels = model_data['labels']
    image = model_data['image']

    go_trial = np.zeros((input_act.shape[2], 19))
    for cell in range(go_trial.shape[0]):
        idx = np.argwhere((labels.squeeze() == 1) &
                          (image == pref_image[cell]))
        go_trials = []
        for trial in idx:
            trial_chunk = input_act[trial[0],
                                    (trial[1]-9):(trial[1]+9+1), cell]
            if trial_chunk.shape[0] == go_trial.shape[1]:
                go_trials.append(trial_chunk)
        go_trial[cell, :] = np.stack(go_trials).mean(axis=0)

    return go_trial


def plot_go_trial(go_trial, ax):
    # fig, ax = plt.subplots(figsize=(3, 2))
    # ax.plot(go_trial.mean(axis=0), color='blue', linewidth=3)
    ax.plot(go_trial.mean(axis=0) / go_trial.mean(axis=0).max(),
            color='blue', linewidth=3)
    ax.set_xticks(np.linspace(0, 18, 3), (-2.25, 0, 2.25))
    ax.set_yticks([0, 0.5, 1.0])
    # ax.set_ylim([0, 1.02])
    ax.set_xlabel('time after change (s)')
    ax.set_ylabel('a.u.')


def get_adapt_mean_slope(go_trial):
    x = np.arange(4)
    # starting at event, 3::
    y = go_trial[:, ::3].mean(axis=0)[3::]
    g, _ = np.polyfit(x, y, 1)
    return g


def get_n_reps(model_data):
    model_labels = model_data['labels']
    # store number of repeats
    n_reps = np.zeros_like(model_labels[:, ::3]).squeeze()
    # iterate over batches
    for b in range(model_labels.shape[0]):
        cnt = -1
        # iterate over flashes
        for i, label in enumerate(model_labels[b, ::3]):
            cnt += 1
            if label == 1:
                cnt = 0
            n_reps[b, i] = cnt

    n_reps += 1
    return n_reps


def get_model_df(model_data):
    n_reps = get_n_reps(model_data)
    data = model_data['hidden']  # choose either input layer or hidden layer
    num_cells = data.shape[2]
    # Get cell flashes (ignore initial set of flashes)
    cell_flash = data[:, 12::3].reshape(-1, num_cells)
    # remove omitted flashes
    img_model = model_data['image'][:, 12::3].flatten()
    cell_flash = cell_flash[img_model != 8]
    # first remove first 12 flashes, then omitted flashes
    n_reps = n_reps[:, 4:].flatten()
    n_reps = n_reps[img_model != 8]
    # remove omitted flashes
    img_model = img_model[img_model != 8]
    # Define mapping between image index and image label
    stim_dict_new = {0: 'im077', 1: 'im062', 2: 'im066', 3: 'im063',
                     4: 'im065', 5: 'im069', 6: 'im085', 7: 'im061', 8: 'blank'}
    # PCA
    pca = PCA(n_components=num_cells)
    X_pca = pca.fit(cell_flash).transform(cell_flash)

    model_df = pd.DataFrame(
        {'image_name': [stim_dict_new[item] for item in img_model], 'repeat': n_reps})
    # fill DF
    for comp in range(3):
        model_df['pca'+str(comp+1)] = X_pca[:, comp]

    model_df['pca_distance'] = np.linalg.norm(
        model_df.loc[:, model_df.columns[2:]].values, axis=1)
    model_df['distance'] = np.linalg.norm(cell_flash, axis=1)
    model_df['pca1_distance'] = np.abs(
        model_df.loc[:, model_df.columns[2]].values)
    model_df['pca2_distance'] = np.abs(
        model_df.loc[:, model_df.columns[3]].values)
    model_df['n_reps'] = n_reps

    return model_df


def plot_stim_pca(model_df, ax):
    ax.scatter(
        model_df['pca1'],
        model_df['pca2'],
        c=pd.factorize(model_df['image_name'])[0],
        cmap='jet',
        alpha=0.5,
    )
    # ax.set_ylim(-4, 8)
    # ax.set_xlim(-10, 10)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    # ax.legend(frameon=False, loc='right', bbox_to_anchor=(1.35, 0.5))


def plot_repeat_pca(model_df, ax):
    ax.scatter(
        model_df['pca1'],
        model_df['pca2'],
        c=model_df['n_reps'],
        cmap='Blues',
    )

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')


def plot_repeat_distances(model_df, ax):
    n_values = 20
    euclidean_distance = model_df.groupby(
        'repeat')['distance'].mean().values[:n_values]
    pca_1_distance = model_df.groupby(
        'repeat')['pca1_distance'].mean().values[:n_values]
    pca_2_distance = model_df.groupby(
        'repeat')['pca2_distance'].mean().values[:n_values]

    ax.plot(np.arange(1, n_values+1), euclidean_distance /
            euclidean_distance.max(), linewidth=4)
    ax.plot(np.arange(1, n_values+1), pca_1_distance /
            pca_1_distance.max(), linewidth=4)
    ax.plot(np.arange(1, n_values+1), pca_2_distance /
            pca_2_distance.max(), linewidth=4)

    ax.set_ylim([0, 1.02])
    ax.set_xticks(np.arange(1, n_values+1, 5))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Repeat')
    ax.set_ylabel('Normalized distance')
    ax.legend(['Euclidean', 'PC1', 'PC2'], frameon=False)


def get_omit_trials(model_data):
    input_act = model_data['input']
    omit = model_data['omit']
    # find omitted trials
    idx = np.argwhere(omit == 1)

    omit_trial = []
    for trial in idx:
        trial_chunk = input_act[trial[0], (trial[1]-9):(trial[1]+9+1)]
        if trial_chunk.shape[0] == 19:
            omit_trial.append(trial_chunk)

    omit_trial = np.stack(omit_trial).mean(axis=0).transpose()
    return omit_trial


def plot_omit_mean(omit_trial, ax):
    ax.plot(omit_trial.mean(axis=0) / omit_trial.mean(axis=0).max(),
            color='blue', linewidth=3)
    # ax.plot(omit_trial.T)
    # ax.plot(omit_trial.mean(axis=0), color='blue', linewidth=3)

    ax.set_xticks(np.linspace(0, 18, 3), (-2.25, 0, 2.25))
    # ax.set_yticks([0, 0.5, 1])
    # ax.set_ylim([0, 1.02])

    ax.set_xlabel('time after omit (s)')
    ax.set_ylabel('a.u.')


def plot_model_tings(result_path):
    model_data, pref_image = get_acts(result_path)
    go_trial = get_go_trials(model_data, pref_image)
    model_df = get_model_df(model_data)
    omit_trial = get_omit_trials(model_data)
    # plot
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    plot_go_trial(go_trial, ax[0, 0])
    # plot_stim_pca(model_df, ax[0, 1])
    plot_repeat_pca(model_df, ax[0, 1])
    plot_repeat_distances(model_df, ax[1, 0])
    plot_omit_mean(omit_trial, ax[1, 1])
