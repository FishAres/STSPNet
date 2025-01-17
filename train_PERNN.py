import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from stimulus import StimGenerator
from models import *
from utilities import *


def PERNN_loss(output, inputs_post, labels, a_hat, rnn_criterion, args):
    L1 = rnn_criterion(output, labels.clamp(min=0))
    L_pred_err = torch.mean((a_hat - inputs_post)**2)
    # L_pred_err = F.binary_cross_entropy(a_hat, inputs_norm)
    return L1, L_pred_err


def train_me(args, device, train_generator, model, optimizer):
    """
    Train model
    """
    model.train()

    # Get inputs and labels
    inputs, inputs_prev, labels, _, mask, _ = train_generator.generate_batch()
    # Send to device
    inputs_post = np.roll(inputs, -1, axis=1)
    inputs_post = inputs_post[:,0:-1,:]

    inputs = torch.from_numpy(inputs).to(device)
    inputs_prev = torch.from_numpy(inputs_prev).to(device)
    inputs_post = torch.from_numpy(inputs_post).to(device)

    labels = torch.from_numpy(labels).to(device)
    mask = torch.from_numpy(mask).to(device)

    # Initialize syn_x or hidden state
    model.syn_x = model.init_syn_x(args.batch_size).to(device)
    model.hidden = model.init_hidden(args.batch_size).to(device)

    optimizer.zero_grad()
    output, hidden, inputs, a_hat = model(inputs_prev)

    # Convert to binary prediction
    output = torch.sigmoid(output)
    pred = torch.bernoulli(output).byte()

    labels = labels[:, 0:-1, :]
    mask = mask[:,0:-1,:]

    # Compute hit rate and false alarm rate
    hit_rate = (pred * (labels == 1)).sum().float().item() / \
        (labels == 1).sum().item()
    fa_rate = (pred * (labels == -1)).sum().float().item() / \
        (labels == -1).sum().item()

    dprime_true = dprime(hit_rate, fa_rate)

    rnn_criterion = torch.nn.BCEWithLogitsLoss(
        reduction='none', pos_weight=torch.tensor([args.pos_weight]).to(device))

    # Clamp to zero since we only want "go" labels
    l_task, l_pred_err = PERNN_loss(output, inputs_post, labels, a_hat, rnn_criterion, args)
    # Apply mask and take mean
    loss = l_task + args.pred_loss_w * l_pred_err
    loss = (loss * mask).mean()
    
    l_task = (l_task * mask).mean()
    l_pred_err = (l_pred_err * mask).mean()

    # L2 loss on hidden unit activations
    L2_loss = hidden.pow(2).mean()
    loss += args.l2_penalty * L2_loss

    loss.backward()
    optimizer.step()

    return loss.item(), l_task.item(), l_pred_err.item(), dprime_true.item()


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
        model.syn_x = model.init_syn_x(args.batch_size).to(device)
        model.hidden = model.init_hidden(args.batch_size).to(device)

        output, hidden, inputs, a_hat = model(inputs)
    # Convert to binary prediction
    output = torch.sigmoid(output)
    pred = torch.bernoulli(output).byte()

    labels = labels[:,0:-1,:]
    mask = mask[:,0:-1,:]

    # Compute hit rate and false alarm rate
    hit_rate = (pred * (labels == 1)).sum().float().item() / \
        (labels == 1).sum().item()
    fa_rate = (pred * (labels == -1)).sum().float().item() / \
        (labels == -1).sum().item()

    # Compute dprime
    dprime_true = dprime(hit_rate, fa_rate)
    
    
    return dprime_true.item(), hit_rate, fa_rate, inputs, hidden, output, pred, image, labels, omit


def train_PERNN():
    # Training settings
    parser = argparse.ArgumentParser(description='Models of change detection')
    parser.add_argument('--image-set', type=str, default='A', metavar='I',
                        help='image set to train on: A, B, C, D (default: A)')
    parser.add_argument('--model', type=str, default='STPNet', metavar='M',
                        help='model to train: STPNet, RNN, or STPRNN (default: STPNet)')
    parser.add_argument('--noise-std', type=float, default=0.0, metavar='N',
                        help='standard deviation of noise (default: 0.0)')
    parser.add_argument('--syn-tau', type=float, default=6.0, metavar='N',
                        help='STPNet recovery time constant (default: 6.0)')
    parser.add_argument('--hidden-dim', type=int, default=16, metavar='N',
                        help='hidden dimension of model (default: 16)')
    parser.add_argument('--l2-penalty', type=float, default=0.0, metavar='L2',
                        help='L2 penalty on hidden activations (default: 0.0)')
    parser.add_argument('--pos-weight', type=float, default=1.0, metavar='W',
                        help='weight on positive examples (default: 1.0)')
    parser.add_argument('--seq-length', type=int, default=50000, metavar='N',
                        help='length of each trial (default: 50000)')
    parser.add_argument('--delay-dur', type=int, default=500, metavar='N',
                        help='delay duration (default: 500 ms)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='number of train trial batches (default: 128)')
    parser.add_argument('--epochs', type=int, default=2500, metavar='N',
                        help='epoch train criterion (default: 5000)')
    parser.add_argument('--dprime', type=float, default=2.0, metavar='N',
                        help='dprime train criterion (default: 2.0)')
    parser.add_argument('--patience', type=int, default=1, metavar='N',
                        help='number of epochs to wait above baseline (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--pred_loss_w', type=float, default=0.0,
                         help='weighting factor for prediction error loss in PERNN')
    parser.add_argument('--task_loss_thresh', type=float, default=0.32,
                         help='task performance criterion to stop training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create train stimulus generator
    train_generator = StimGenerator(image_set=args.image_set, seed=args.seed,
                                    batch_size=args.batch_size, seq_length=args.seq_length,
                                    delay_dur=args.delay_dur)

    # Get input dimension of feature vector
    input_dim = len(train_generator.feature_dict[0])

    # Create model
    if args.model in ['PERNN', 'PERNN_sub', 'PERNN_sub_STP']:
        model = PERNN(input_dim=input_dim,
                      hidden_dim=args.hidden_dim,
                      syn_tau=args.syn_tau,
                      noise_std=args.noise_std).to(device)
    else:
        raise ValueError("Model not found")

    # Define loss function
    # criterion = torch.nn.BCEWithLogitsLoss(
        # reduction='none', pos_weight=torch.tensor([args.pos_weight]).to(device))

    optimizer = torch.optim.Adam(model.parameters())

    # Initialize tracking variables
    loss_list = []
    l_task_list = []
    l_pred_err_list = []
    dprime = 0
    dprime_list = []
    wait = 0

    for epoch in range(1, args.epochs + 1):
        # Train model
        loss, l_task, l_pred_err, dprime = train_me(args, device, train_generator,
                                model, optimizer)

        loss_list.append(loss)
        l_task_list.append(l_task)
        l_pred_err_list.append(l_pred_err)
        dprime_list.append(dprime)

        if epoch % args.log_interval == 0:
            # Print current progress
            print("Epoch: {}  task loss: {:.4f} pred loss: {:.4f} dprime: {:.2f}".format(
                epoch, l_task, l_pred_err, dprime))

        if l_task < args.task_loss_thresh:
            if dprime < args.dprime:
                # Reset wait count
                wait = 0
            else:
                # Increase wait count
                wait += 1
                # Stop training after wait exceeds patience
                if wait >= args.patience:
                    break

    # Save trained model
    save_dir = './PARAM/'+args.model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir+'/model_train_seed_'+str(args.seed)+'_' + str(args.pred_loss_w)+'.pt'
    save_path = save_dir+'/model_train_seed_'+str(args.seed)+'_' + str(args.pred_loss_w)+'.pt'
    torch.save({'epoch': epoch,
                'loss': loss_list,
                'dprime': dprime_list,
                'l_task': l_task_list,
                'l_pred_err': l_pred_err_list,
                'state_dict': model.state_dict()}, save_path)


# if __name__ == 'train_PERNN.py':
train_PERNN()
