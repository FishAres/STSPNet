import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt


def get_hit_false_rate(pred, labels):
    # Compute hit rate and false alarm rate
    hit_rate = (pred * (labels == 1)).sum().float().item() / \
        (labels == 1).sum().item()
    fa_rate = (pred * (labels == -1)).sum().float().item() / \
        (labels == -1).sum().item()
    return hit_rate, fa_rate


def train(args, device, train_generator, model, criterion, optimizer):
    """
    Train model
    """
    model.train()

    # Get inputs and labels
    inputs, inputs_prev, labels, _, mask, _ = train_generator.generate_batch()
    # Send to device
    inputs = torch.from_numpy(inputs).to(device)
    inputs_prev = torch.from_numpy(inputs_prev).to(device)
    labels = torch.from_numpy(labels).to(device)
    mask = torch.from_numpy(mask).to(device)

    # Initialize syn_x or hidden state
    if args.model == 'STPNet' or args.model == 'STPRNN' or args.model == 'STPENet':
        model.syn_x = model.init_syn_x(args.batch_size).to(device)
    if args.model == 'RNN' or args.model == 'STPRNN' or args.model == 'PERNN2':
        model.hidden = model.init_hidden(args.batch_size).to(device)

    optimizer.zero_grad()
    if args.model == 'PERNN':
        output, hidden, _, a_hat = model(inputs, inputs_prev)
    elif args.model == 'STPNet' or args.model == 'RNN' or args.model == 'STPRNN':
        output, hidden, _ = model(inputs)
    elif args.model == 'PERNN2':
        output, hidden, _ = model(inputs, inputs_prev)
    else:
        output, hidden, _ = model(inputs, inputs_prev)

    # Convert to binary prediction
    output = torch.sigmoid(output)
    pred = torch.bernoulli(output).byte()

    # Compute hit rate and false alarm rate
    hit_rate, fa_rate = get_hit_false_rate(pred, labels)
    # Compute dprime
    dprime_true = dprime(hit_rate, fa_rate)

    loss = criterion(output, labels.clamp(min=0))
    # Apply mask and take mean
    loss = (loss * mask).mean()

    # L2 loss on hidden unit activations
    L2_loss = hidden.pow(2).mean()
    loss += args.l2_penalty * L2_loss

    loss.backward()
    optimizer.step()

    return loss.item(), dprime_true.item()


def test(args, device, test_generator, model):
    """
    Test model, get predictions and plot confusion matrix
    """
    model.eval()

    with torch.no_grad():
        # Get inputs and labels
        inputs, inputs_prev, labels, image,  _, omit = test_generator.generate_batch()

        # Send to device
        inputs = torch.from_numpy(inputs).to(device)
        inputs_prev = torch.from_numpy(inputs_prev).to(device)
        labels = torch.from_numpy(labels).to(device)

        # Initialize syn_x or hidden state
        if args.model == 'STPNet' or args.model == 'STPRNN' or args.model == 'STPENet':
            model.syn_x = model.init_syn_x(args.batch_size).to(device)
        if args.model == 'RNN' or args.model == 'STPRNN' or args.model == 'PERNN2':
            model.hidden = model.init_hidden(args.batch_size).to(device)

        if args.model == 'STPNet' or args.model == 'RNN' or args.model == 'STPRNN':
            output, hidden, input_syn = model(inputs)
        else:
            output, hidden, input_syn = model(inputs, inputs_prev)
        # output, hidden, inputs, input_syn = model(inputs)  # for visualization below

    # Convert to binary prediction
    output = torch.sigmoid(output)
    pred = torch.bernoulli(output).byte()

    # Compute hit rate and false alarm rate
    hit_rate = (pred * (labels == 1)).sum().float().item() / \
        (labels == 1).sum().item()
    fa_rate = (pred * (labels == -1)).sum().float().item() / \
        (labels == -1).sum().item()

    # Compute dprime
    # dprime_true = dprime(hit_rate, fa_rate)
    go = (labels == 1).sum().item()
    catch = (labels == -1).sum().item()
    num_trials = (labels != 0).sum().item()
    assert (go + catch) == num_trials

    # dprime_true = compute_dprime(hit_rate, fa_rate, go, catch, num_trials)
    # dprime_old = dprime(hit_rate, fa_rate)
    dprime_true = dprime(hit_rate, fa_rate)
    # try:
    #     assert dprime_true == dprime_old
    # except:
    #     print(hit_rate, fa_rate)
    #     print(dprime_true, dprime_old)

    return dprime_true.item(), hit_rate, fa_rate, input_syn, hidden, output, pred, image, labels, omit


def trial_number_limit(p, N):
    if N == 0:
        return np.nan
    else:
        p = np.max((p, 1. / (2 * N)))
        p = np.min((p, 1 - 1. / (2 * N)))
    return p


def compute_dprime(hit_rate, fa_rate, go_trials, catch_trials, total_trials):
    """ calculates the d-prime for a given hit rate and false alarm rate
    https://en.wikipedia.org/wiki/Sensitivity_index
    Parameters
    ----------
    hit_rate : float
    rate of hits in the True class
    fa_rate : float
    rate of false alarms in the False class
    go_trials: int
    number of go trials
    catch_trials: int
    number of catch trials
    total_trials: int
    total number of trials
    Returns
    -------
    d_prime
    """
    limits = (1./total_trials, 1 - 1./total_trials)
    assert limits[0] > 0.0, 'limits[0] must be greater than 0.0'
    assert limits[1] < 1.0, 'limits[1] must be less than 1.0'
    assert (go_trials + catch_trials) == total_trials
    Z = norm.ppf
    # Limit values in order to avoid d' infinity
    hit_rate = np.clip(trial_number_limit(
        hit_rate, go_trials), limits[0], limits[1])
    fa_rate = np.clip(trial_number_limit(
        fa_rate, catch_trials), limits[0], limits[1])

    return Z(hit_rate) - Z(fa_rate)


def dprime(hit_rate, fa_rate, limits=(0.01, 0.99)):
    """ calculates the d-prime for a given hit rate and false alarm rate
    https://en.wikipedia.org/wiki/Sensitivity_index
    Parameters
    ----------
    hit_rate : float
    rate of hits in the True class
    fa_rate : float
    rate of false alarms in the False class
    limits : tuple, optional
    limits on extreme values, which distort. default: (0.01,0.99)
    Returns
    -------
    d_prime
    """
    assert limits[0] > 0.0, 'limits[0] must be greater than 0.0'
    assert limits[1] < 1.0, 'limits[1] must be less than 1.0'
    Z = norm.ppf
    # Limit values in order to avoid d' infinity
    hit_rate = np.clip(hit_rate, limits[0], limits[1])
    fa_rate = np.clip(fa_rate, limits[0], limits[1])
    return Z(hit_rate) - Z(fa_rate)


def compute_confusion_matrix(num_images,
                             labels,
                             image,
                             output,
                             step=3,
                             plot=False,
                             image_ticks=None,
                             matrix_plot_save_path=None):
    """
    Compute confusion matrix
    Plot and save confusion matrix if needed
    """
    # Initialize confusion matrix
    response_matrix = np.zeros((num_images, num_images))
    total_matrix = np.zeros((num_images, num_images))

    switch = (labels != 0)
    switch = switch.flatten()
    image = image.flatten()
    output = output.flatten()

    for i in range(len(image)):
        if switch[i] == 1:  # if there is a switch
            new_img = image[i]
            old_img = image[i-step]
            total_matrix[old_img, new_img] += 1
            if (output[i] == 1):  # need to modify to depend on target window
                response_matrix[old_img, new_img] += 1

    confusion_matrix = response_matrix / total_matrix

    if plot:
        plt.figure()
        plt.imshow(confusion_matrix, cmap='magma', vmin=0, vmax=1)
        plt.colorbar()

        if image_ticks is not None:
            plt.xticks(np.arange(8), image_ticks)
            plt.yticks(np.arange(8), image_ticks)

        plt.title('Response Probability Matrix')
        plt.xlabel('Initial Image')
        plt.ylabel('Change Image')

        if matrix_plot_save_path is not None:
            plt.savefig(matrix_plot_save_path + '.png', bbox_inches="tight")

        # plt.close()

    return response_matrix, total_matrix, confusion_matrix
