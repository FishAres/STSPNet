import os
import argparse
import numpy as np
import torch

from scipy.stats import truncexpon
# from stimulus import StimGenerator
from models import STPNet, OptimizedRNN, STPRNN
from utilities import *


class StimGenerator:
    def __init__(self,
                 image_set='A',       # image set ('A', 'B', 'C', or 'D')
                 seq_length=50000,    # length of each "trial" (ms)
                 batch_size=128,      # number of "trial" batches
                 time_step=250,       # simulation time step (ms)
                 image_pres_dur=250,  # image presentation (ms)
                 delay_dur=500,       # delay duration (ms)
                 reward_on=0,         # reward window (ms)
                 reward_off=250,
                 rep_min=4,           # minimum number of repeats (inclusive)
                 rep_max=12,          # maximum number of repeats (exclusive)
                 omit_frac=0.0,       # fraction of omitted flashes
                 seed=1):             # random seed

        self.image_set = image_set
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.time_step = time_step
        self.image_steps = image_pres_dur // time_step
        self.delay_steps = delay_dur // time_step
        self.reward_on_steps = reward_on // time_step
        self.reward_off_steps = reward_off // time_step
        self.min_repeat = rep_min
        self.max_repeat = rep_max
        self.omit_frac = omit_frac
        self.seed = seed

        # Get image features
        self.feature_dict, self.num_images = self.load_image_features()

        # Number of steps to mask loss
        self.mask_steps = 4*(self.image_steps + self.delay_steps)

    def load_image_features(self):
        """
        Loads feature dict for a given image set.
        """
        if self.image_set == 'A':
            feature_dict = np.load(
                './cifar/features/image_set_cnn_A_seed_'+str(self.seed)+'.npy')
            img_ind_swap = np.array([1, 5, 2, 4, 0, 7, 6, 3])
            # image_ticks = ('077', '062', '066', '063', '065',
            #                '069', '085', '061')
        elif self.image_set == 'B':
            feature_dict = np.load(
                './cifar/features/image_set_cnn_B_seed_'+str(self.seed)+'.npy')
            img_ind_swap = np.array([0, 5, 7, 1, 3, 6, 4, 2])
            # image_ticks = ('012', '057', '078', '013', '047',
            #                '036', '044', '115')
        elif self.image_set == 'C':
            feature_dict = np.load(
                './cifar/features/image_set_cnn_C_seed_'+str(self.seed)+'.npy')
            img_ind_swap = np.array([3, 2, 6, 1, 5, 7, 4, 0])
            # image_ticks = ('073', '075', '031', '106', '054',
            #                '035', '045', '000')
        elif self.image_set == 'D':
            feature_dict = np.load(
                './cifar/features/image_set_cnn_D_seed_'+str(self.seed)+'.npy')
            img_ind_swap = np.array([3, 1, 5, 4, 7, 6, 2, 0])
            # image_ticks = ('072', '114', '091', '087', '034',
            #                '024', '104', '005')

        # Resort by image detectability
        feature_dict[:-1, :] = feature_dict[img_ind_swap, :]

        return feature_dict, len(feature_dict)-1

    def _generate_num_repeat(self, scale=2.0):
        """
        Helper function, randomly generate number of repetition for an image.
        Choose an integer with exponential distribution between [min, max].

        Output:
          integer between [min_repeat, max_repeat)
        """
        min_switch = truncexpon.rvs(b=(self.max_repeat - self.min_repeat) /
                                    scale, loc=self.min_repeat, scale=scale).astype(int)

        return min_switch

    def generate_batch(self):
        """
        Generate one batch of inputs

        Args:
            batch_len: number of trials per batch
            feature_dict: list of features
        """
        # Initialize output arrays
        image_array = np.zeros(
            (self.batch_size, self.seq_length // self.time_step), dtype='int')
        label_array = np.zeros(
            (self.batch_size, self.seq_length // self.time_step, 1), dtype='float32')
        mask_array = np.zeros(
            (self.batch_size, self.seq_length // self.time_step, 1), dtype='float32')
        # Mask out blank flashes
        mask_array[:, ::(self.image_steps+self.delay_steps)] = 1

        # Loop over each element in batch
        for i in range(self.batch_size):
            last_image = -1
            image = np.array([], dtype='int')
            while len(image) < (self.seq_length // self.time_step):
                # Generate random image and number of repeats
                image_num = np.random.randint(self.num_images)
                repeat_num = self._generate_num_repeat()

                image_repeat = np.tile(
                    [image_num]*self.image_steps+[self.num_images]*self.delay_steps, repeat_num)

                if image_num != last_image:
                    # Go trial
                    label_array[i, len(
                        image)+self.reward_on_steps:len(image)+self.reward_off_steps] = 1
                    last_image = image_num
                else:
                    # Catch trial
                    label_array[i, len(image)] = -1

                image = np.concatenate((image, image_repeat))

            # Use only seq_length
            image_array[i, :] = image[:(self.seq_length // self.time_step)]
            input_array = self.feature_dict[image_array, :]

        # Omitted flashes
        if self.omit_frac > 0:
            pad = self.image_steps + self.delay_steps
            omit_array = (np.random.binomial(1, self.omit_frac, size=image_array.shape)) & \
                (image_array != self.num_images) & (label_array.squeeze() == 0) & \
                (np.pad(label_array.squeeze(), ((0, 0), (0, pad)),
                        mode='constant')[:, pad:] == 0)

            # Re-assign omitted flashes here
            image_array[np.where(omit_array)[0], np.where(
                omit_array)[1]] = self.num_images
            input_array[np.where(omit_array)[0], np.where(
                omit_array)[1]] = self.feature_dict[-1, :]
        else:
            omit_array = np.zeros(
                (self.batch_size, self.seq_length // self.time_step), dtype='int')

        # Set first image to always be zero
        label_array[:, :self.reward_off_steps] = 0

        # Mask out first part of stimulus
        mask_array[:, :self.mask_steps] = 0

        return input_array, label_array, image_array, mask_array, omit_array

def test(args, device, test_generator, model):
    """
    Test model, get predictions and plot confusion matrix
    """
    model.eval()

    with torch.no_grad():
        # Get inputs and labels
        inputs, labels, image,  _, omit = test_generator.generate_batch()

        # Send to device
        inputs = torch.from_numpy(inputs).to(device)
        labels = torch.from_numpy(labels).to(device)

        # Initialize syn_x or hidden state
        if args.model == 'STPNet' or args.model == 'STPRNN':
            model.syn_x = model.init_syn_x(args.batch_size).to(device)
        if args.model == 'RNN' or args.model == 'STPRNN':
            model.hidden = model.init_hidden(args.batch_size).to(device)

        output, hidden, input_syn = model(inputs)

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
    elif args.model == 'STPRNN':
        model = STPRNN(input_dim=input_dim,
                       hidden_dim=args.hidden_dim,
                       syn_tau=args.syn_tau,
                       noise_std=args.noise_std).to(device)
    elif args.model == 'RNN':
        model = OptimizedRNN(input_dim=input_dim,
                             hidden_dim=args.hidden_dim,
                             noise_std=args.noise_std).to(device)
    else:
        raise ValueError("Model not found")

    # Load saved parameters
    model.load_state_dict(torch.load(args.model_path)['state_dict'])

    # Test model
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
    response_matrix, total_matrix, confusion_matrix = compute_confusion_matrix(test_generator.num_images, labels,
                                                                               image, pred,
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
        [args.model, args.image_set, str(args.seed)])+'.pkl'), 'wb'), protocol=2)


if __name__ == '__main__':
    main()