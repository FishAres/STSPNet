import os
import argparse
import numpy as np
import torch

from stimulus import StimGenerator
from models import *
from utilities import *  # compute_confusion_matrix


def test_PERNN2(args, device, test_generator, model):
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
        model.syn_x = model.init_syn_x(args.batch_size).to(device)
        model.hidden = model.init_hidden(args.batch_size).to(device)

        output, hidden, inputs = model(inputs, inputs_prev)
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

    return dprime_true.item(), hit_rate, fa_rate, inputs, hidden, output, pred, image, labels, omit

def test_PERNN(args, device, test_generator, model):
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
        model.syn_x = model.init_syn_x(args.batch_size).to(device)
        model.hidden = model.init_hidden(args.batch_size).to(device)

        output, hidden, inputs, a_hat = model(inputs)
    # Convert to binary prediction
    output = torch.sigmoid(output)
    pred = torch.bernoulli(output).byte()

    pred = torch.hstack((pred, pred[:,-2:-1,:]))
    
    

    # Compute hit rate and false alarm rate
    hit_rate = (pred * (labels == 1)).sum().float().item() / \
        (labels == 1).sum().item()
    fa_rate = (pred * (labels == -1)).sum().float().item() / \
        (labels == -1).sum().item()

    # Compute dprime
    dprime_true = dprime(hit_rate, fa_rate)
    
    return dprime_true.item(), hit_rate, fa_rate, inputs, hidden, output, pred, image, labels, omit


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Models of change detection')
    parser.add_argument('--image-set', type=str, default='A', metavar='I',
                        help='image set to train on: A, B, C, D (default: A)')
    parser.add_argument('--model', type=str, default='STPNet', metavar='M',
                        help='model to train: STPNet, RNN, or STPRNN (default: STPNet)')
    parser.add_argument('--model-path', type=str, default='',
                        help='path to saved model')
    parser.add_argument('--noise-std', type=float, default=0.0, metavar='N',
                        help='standard deviation of noise (default: 0.0)')
    parser.add_argument('--syn-tau', type=float, default=6.0, metavar='N',
                        help='STPNet recovery time constant (default: 6.0)')
    parser.add_argument('--hidden-dim', type=int, default=16, metavar='N',
                        help='hidden dimension of model (default: 16)')
    parser.add_argument('--seq-length', type=int, default=50000, metavar='N',
                        help='length of each trial (default: 50000)')
    parser.add_argument('--delay-dur', type=int, default=500, metavar='N',
                        help='delay duration (default: 500 ms)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='number of test trial batches (default: 128)')
    parser.add_argument('--omit-frac', type=float, default=0.0, metavar='O',
                        help='fraction of omitted flashes (default: 0.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--pred_loss_w', type=float, default=0.0)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create test stimulus generator
    test_generator = StimGenerator(image_set=args.image_set, seed=args.seed,
                                   batch_size=args.batch_size, seq_length=args.seq_length,
                                   delay_dur=args.delay_dur, omit_frac=args.omit_frac)


    # Get input dimension of feature vector
    input_dim = len(test_generator.feature_dict[0])

    # Create model
    if args.model == 'STPNet':
        model = STPNet(input_dim=input_dim,
                       hidden_dim=args.hidden_dim,
                       syn_tau=args.syn_tau,
                       noise_std=args.noise_std).to(device)
    elif args.model == 'STPENet':
        model = STPENet(input_dim=input_dim,
                        hidden_dim=args.hidden_dim,
                        syn_tau=args.syn_tau,
                        noise_std=args.noise_std).to(device)
    elif args.model == 'STPRNN':
        model = STPRNN(input_dim=input_dim,
                       hidden_dim=args.hidden_dim,
                       syn_tau=args.syn_tau,
                       noise_std=args.noise_std).to(device)
    elif args.model == 'RNN':
        model = OptimizedRNN(input_dim=input_dim,
                             hidden_dim=args.hidden_dim,
                             noise_std=args.noise_std).to(device)
    elif args.model == 'PERNN2':
        model = PERNN2(input_dim=input_dim,
                       hidden_dim=args.hidden_dim,
                       noise_std=args.noise_std).to(device)
    elif args.model == 'PERNN':
        model = PERNN(input_dim=input_dim,
                       hidden_dim=args.hidden_dim,
                       noise_std=args.noise_std).to(device)
    elif args.model == 'PERNN_sub':
        model = PERNN_sub(input_dim=input_dim,
                       hidden_dim=args.hidden_dim,
                       noise_std=args.noise_std).to(device)
    elif args.model == 'PERNN_sub_STP':
        model = PERNN_sub_STP(input_dim=input_dim,
                       hidden_dim=args.hidden_dim,
                       noise_std=args.noise_std).to(device)
    else:
        raise ValueError("Model not found")

    # Load saved parameters
    model.load_state_dict(torch.load(args.model_path)['state_dict'])

    # Test model
    if args.model == 'PERNN2':
        dprime, hr, far, input, hidden, output, pred, image, labels, omit = test_PERNN2(
            args, device, test_generator, model)
    elif args.model == 'PERNN' or args.model == 'PERNN_sub':
        dprime, hr, far, input, hidden, output, pred, image, labels, omit = test_PERNN(
            args, device, test_generator, model)
    else:
        dprime, hr, far, input, hidden, output, pred, image, labels, omit = test(
            args, device, test_generator, model)

    # Save results
    results_dict = {}
    results_dict['dprime'] = dprime
    results_dict['hr'] = hr
    results_dict['far'] = far
    results_dict['input'] = input.cpu().numpy()
    results_dict['hidden'] = hidden.cpu().numpy()
    results_dict['output'] = output.cpu().numpy()
    results_dict['pred'] = pred.cpu().numpy()
    results_dict['image'] = image
    results_dict['labels'] = labels.cpu().numpy()

    # Compute confusion matrix
    response_matrix, total_matrix, confusion_matrix = compute_confusion_matrix(
    test_generator.num_images, labels, image, pred,
    test_generator.image_steps+test_generator.delay_steps)

    results_dict['response_matrix'] = response_matrix
    results_dict['total_matrix'] = total_matrix
    results_dict['confusion_matrix'] = confusion_matrix

    # Compute omitted flash results
    if args.omit_frac > 0:
        shift = 3
        results_dict['omit'] = omit

        all_flashes = np.where(
            (labels.cpu().numpy().squeeze() == 0) & (image != 8) & (omit == 0))
        omitted_flashes = np.where(omit)
        post_omitted_flashes = np.where(
            np.pad(omit, ((0, 0), (shift, 0)), mode='constant')[:, :-shift])

        results_dict['all_flashes'] = (pred[all_flashes[0],
                                            all_flashes[1]].sum().float() / len(all_flashes[0])).item()
        results_dict['omitted_flashes'] = (pred[omitted_flashes[0],
                                                omitted_flashes[1]].sum().float() / len(omitted_flashes[0])).item()
        results_dict['post_omitted_flashes'] = (pred[post_omitted_flashes[0],
                                                     post_omitted_flashes[1]].sum().float() / len(post_omitted_flashes[0])).item()

    import pickle
    save_path = './RESULT/'+args.model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(results_dict, open(os.path.join(save_path, "_".join(
        [args.model, args.image_set, str(args.seed)]) + '_' + str(args.pred_loss_w) + '.pkl'), 'wb'), protocol=2)


if __name__ == '__main__':
    main()
