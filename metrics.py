import numpy as np
import time
import sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# print(plt.style.available)
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import scipy.stats as stats
from scipy.spatial.distance import cdist, directed_hausdorff, jensenshannon
from sewar.full_ref import msssim
import plotting


def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.zeros(size=(labels.size(0), C, labels.size(2), labels.size(3)), dtype=torch.uint8, device=labels.device)
    target = one_hot.scatter_(dim=1, index=labels.to(torch.long), src=torch.tensor(1, dtype=torch.uint8, device=labels.device))

    # target = Variable(target)

    return target


def one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels.to(torch.long)]


def one_hot_numpy(labels, num_classes):
    return np.eye(num_classes)[labels.astype(np.int32)]


def cdist_pytorch(x, y):
    """
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = differences.pow_(2).sum(dim=-1).float().sqrt()
    return distances


def get_confusion_matrix(x, y, mode='single_label', threshold=0.5):  # or mode='multi_label'
    num_classes = x.shape[-1]

    if mode == 'single_label':
        x = torch.argmax(x, dim=-1)
        confusion_matrix = torch.zeros(size=(num_classes, num_classes), dtype=torch.int32)
        for target, prediction in zip(y.flatten(), x.flatten()):
            confusion_matrix[target, prediction] += 1

    elif mode == 'multi_label':
        x = torch.where(x > threshold, torch.ones_like(x), torch.zeros_like(x)).long()
        confusion_matrix = torch.zeros(size=(num_classes, 2, 2), dtype=torch.int32)

        for n_cls in range(num_classes):
            for target, prediction in zip(y[:, n_cls].flatten(), x[:, n_cls].flatten()):
                confusion_matrix[n_cls, target, prediction] += 1

    return confusion_matrix


def get_scores_from_confusion_matrix_multi_label(cm, append_means=True):
    num_classes = cm.shape[0]
    if append_means:
        num_classes += 1

    sensitivities = np.zeros(shape=(num_classes, ), dtype=np.float32)
    specificities = np.zeros(shape=(num_classes, ), dtype=np.float32)
    precisions = np.zeros(shape=(num_classes, ), dtype=np.float32)
    accuracies = np.zeros(shape=(num_classes, ), dtype=np.float32)
    F1_scores = np.zeros(shape=(num_classes, ), dtype=np.float32)

    for i, binary_cm in enumerate(cm):
        sensitivities[i], specificities[i], precisions[i], accuracies[i], F1_scores[i] = get_scores_from_confusion_matrix_binary(binary_cm)

    if append_means:
        sensitivities[-1] = sensitivities[:-1].mean()
        specificities[-1] = specificities[:-1].mean()
        precisions[-1] = precisions[:-1].mean()
        accuracies[-1] = accuracies[:-1].mean()
        F1_scores[-1] = F1_scores[:-1].mean()

    return sensitivities, specificities, precisions, accuracies, F1_scores


def get_scores_from_confusion_matrix_binary(cm):
    eps = 1e-8

    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    sensitivity = TP / np.clip((TP + FN), eps, None)
    specificity = TN / np.clip((TN + FP), eps, None)
    precision = TP / np.clip((TP + FP), eps, None)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    F1_score = 2 * precision * sensitivity / np.clip((precision + sensitivity), eps, None)

    return sensitivity, specificity, precision, accuracy, F1_score


def get_scores_from_confusion_matrix_single_label(cm, append_means=True):
    num_classes = cm.shape[0]
    num_iterations = num_classes
    eps = 1e-8

    if append_means:
        num_classes += 1

    sensitivities = np.zeros(shape=(num_classes, ), dtype=np.float32)
    specificities = np.zeros(shape=(num_classes, ), dtype=np.float32)
    precisions = np.zeros(shape=(num_classes, ), dtype=np.float32)
    accuracies = np.zeros(shape=(num_classes, ), dtype=np.float32)

    for k in range(num_iterations):
        TP = cm[k, k]
        TN = np.delete(np.delete(cm, k, axis=0), k, axis=1).sum()
        FP = np.delete(cm[:, k], k, axis=0).sum()
        FN = np.delete(cm[k, :], k, axis=0).sum()

        sensitivities[k] = TP / np.clip((TP + FN), eps, None)
        specificities[k] = TN / np.clip((TN + FP), eps, None)
        precisions[k] = TP / np.clip((TP + FP), eps, None)
        accuracies[k] = (TP + TN) / (TP + TN + FP + FN)

    F1_scores = 2 * precisions * sensitivities / np.clip((precisions + sensitivities), eps, None)

    if append_means:
        sensitivities[-1] = sensitivities[:-1].mean()
        specificities[-1] = specificities[:-1].mean()
        precisions[-1] = precisions[:-1].mean()
        accuracies[-1] = accuracies[:-1].mean()
        F1_scores[-1] = F1_scores[:-1].mean()

    return sensitivities, specificities, precisions, accuracies, F1_scores


def get_ROC_AUC(labels_pred, labels_true):
    auc = sklearn.metrics.roc_auc_score(labels_true, labels_pred)
    return auc


def get_ROC(labels_pred, labels_true):
    roc = sklearn.metrics.roc_curve(labels_true, labels_pred)
    return roc


def get_AUC(x, y):
    auc = sklearn.metrics.auc(x, y)
    return auc
    

def get_boundary(img):
    # only for binary images
    # checks which highlighted pixels do not have 4 neighboring highlighted pixels
    # returns image with boundary pixels >= 1. Non-boundary pixels are zero. Pixels outside segmented area are zero.
    new = np.copy(img)
    new[:img.shape[0] - 1, :] += img[1:, :]
    new[1:, :] += img[:img.shape[0] - 1, :]
    new[:, :img.shape[0] - 1] += img[:, 1:]
    new[:, 1:] += img[:, :img.shape[0] - 1]
    return img * (5 - new)

#####################
### BASIC METRICS ###
#####################


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        rmse = torch.sqrt(F.mse_loss(input, target))
        return rmse


class L1_Distance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        l1_distance = torch.mean(torch.abs(input - target))

        return l1_distance


###############################
### IMAGE STRUCTURE METRICS ###
###############################

class Q_Index(nn.Module):
    """
    Quality measurement between perturbated (image with applied noise) and denoised target image.
    This module works only for images with a single color channel (grayscale)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        batch_size = input.shape[0]
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        input_mean = input.mean(dim=-1)
        target_mean = target.mean(dim=-1)
        input_var = input.var(dim=-1)
        target_var = target.var(dim=-1)
        mean_inp_times_tar = torch.mean(input * target, dim=-1)  # E(XY)
        covariance = mean_inp_times_tar - input_mean * target_mean  # Cov(X,Y) = E(XY) - E(X)E(Y)

        # input -= input_mean
        # target -= target_mean
        # covariance
        # covariance = torch.mean(input * target)

        Q = (4. * covariance * input_mean * target_mean) / ((input_var + target_var) * (input_mean ** 2 + target_mean ** 2))

        Q = Q.mean()

        return Q


def get_SSIM(imgs_1, imgs_2, return_individual_results=False):
    assert len(imgs_1) == len(imgs_2), 'imgs_1 and imgs_2 must have the same length.'

    ssim_array = np.empty(shape=(len(imgs_1), ), dtype=np.float32)
    for i, (img_1, img_2) in enumerate(zip(imgs_1, imgs_2)):
        print('SSIM iteration {0}/{1}'.format(i + 1, len(ssim_array)), end='\r')
        ssim_array[i] = ssim(img_1, img_2)
    print('' * 100, end='\r')

    ssim_mean = np.mean(ssim_array, axis=0)
    ssim_std = np.std(ssim_array, axis=0)

    if return_individual_results:
        return ssim_mean, ssim_std, ssim_array
    else:
        return ssim_mean, ssim_std


def get_MS_SSIM(imgs_1, imgs_2, return_individual_results=False):
    assert len(imgs_1) == len(imgs_2), 'imgs_1 and imgs_2 must have the same length.'

    ms_ssim_array = np.empty(shape=(len(imgs_1), ), dtype=np.float32)
    for i, (img_1, img_2) in enumerate(zip(imgs_1, imgs_2)):
        print('MS-SSIM iteration {0}/{1}'.format(i + 1, len(ms_ssim_array)), end='\r')
        ms_ssim_array[i] = msssim(img_1, img_2, MAX=1.)
    print('' * 100, end='\r')

    ms_ssim_mean = np.mean(ms_ssim_array, axis=0)
    ms_ssim_std = np.std(ms_ssim_array, axis=0)

    if return_individual_results:
        return ms_ssim_mean, ms_ssim_std, ms_ssim_array
    else:
        return ms_ssim_mean, ms_ssim_std


def get_PSNR(imgs_1, imgs_2, return_individual_results=False):
    assert len(imgs_1) == len(imgs_2), 'imgs_1 and imgs_2 must have the same length.'

    psnr_array = np.empty(shape=(len(imgs_1), ), dtype=np.float32)
    for i, (img_1, img_2) in enumerate(zip(imgs_1, imgs_2)):
        print('PSNR iteration {0}/{1}'.format(i + 1, len(psnr_array)), end='\r')
        psnr_array[i] = psnr(img_1, img_2)
    print('' * 100, end='\r')

    psnr_mean = np.mean(psnr_array, axis=0)
    psnr_std = np.std(psnr_array, axis=0)

    if return_individual_results:
        return psnr_mean, psnr_std, psnr_array
    else:
        return psnr_mean, psnr_std


def get_MSE(imgs_1, imgs_2, return_individual_results=False, get_rmse=False):
    assert len(imgs_1) == len(imgs_2), 'imgs_1 and imgs_2 must have the same length.'

    mse_array = np.empty(shape=(len(imgs_1), ), dtype=np.float32)
    for i, (img_1, img_2) in enumerate(zip(imgs_1, imgs_2)):
        print('MSE iteration {0}/{1}'.format(i + 1, len(mse_array)), end='\r')
        if get_rmse:
            mse_array[i] = np.sqrt(psnr(img_1, img_2))
        else:
            mse_array[i] = mse(img_1, img_2)
    print('' * 100, end='\r')

    mse_mean = np.mean(mse_array, axis=0)
    mse_std = np.std(mse_array, axis=0)

    if return_individual_results:
        return mse_mean, mse_std, mse_array
    else:
        return mse_mean, mse_std


def get_Q(imgs_1, imgs_2, return_individual_results=False):

    imgs_1_mean = np.mean(imgs_1, axis=(-2, -1))
    imgs_2_mean = np.mean(imgs_2, axis=(-2, -1))
    imgs_1_var = np.var(imgs_1, axis=(-2, -1))
    imgs_2_var = np.var(imgs_2, axis=(-2, -1))

    mean_imgs_1_times_imgs_2 = np.mean(imgs_1 * imgs_2, axis=(-2, -1))  # E(XY)
    covariance = mean_imgs_1_times_imgs_2 - imgs_1_mean * imgs_2_mean  # Cov(X,Y) = E(XY) - E(X)E(Y)

    Q = (4. * covariance * imgs_1_mean * imgs_2_mean) / ((imgs_1_var + imgs_2_var) * (imgs_1_mean ** 2 + imgs_2_mean ** 2))

    return Q.mean(), Q.std()


def get_JS_divergence(imgs_1, imgs_2, masks_1, masks_2, num_classes, mode='255', remove_till_val=None, get_distance=False):
    """
    Calculate JS divergence for masked image regions.
    imgs_1, imgs_2 (ndarray):   Grayscale-image arrays with shapes [n, h, w] and grayvalues in [0 ... 1]
    masks_1, masks_2 (ndarray): Mask arrays with shape [n, h, w] and values in {0, 1, ..., num_classes - 1}
    num_classes (int):          Number of segmentation classen
    mode (string):              '255' or None. Only important, if less classes than 256
    remove_till_val (float):    Grayscale values smaller this value are not considered for histograms and JS-divergence
    get_distance (bool):        If true, squareroot of JS-divergence is returned (distance)
    """
    assert masks_1.shape == masks_2.shape == imgs_1.shape == imgs_2.shape, 'All input arrays must have the same shape.'

    if mode == '255':
        imgs_1 = (imgs_1 * 255).astype(np.int32)
        imgs_2 = (imgs_2 * 255).astype(np.int32)
        bins = np.arange(0, 256, 1)
        if remove_till_val is not None:
            remove_till_val = remove_till_val * 255
    else:
        bins = np.linspace(0, 1, 257, endpoint=True)

    js_divs = np.empty(shape=(num_classes, ), dtype=np.float32)
    for i in range(num_classes):
        imgs_1_masked = imgs_1[masks_1 == i]
        imgs_2_masked = imgs_2[masks_2 == i]

        if remove_till_val is not None:
            imgs_1_masked = imgs_1_masked[imgs_1_masked > remove_till_val]
            imgs_2_masked = imgs_2_masked[imgs_2_masked > remove_till_val]

        """
        params_1 = scipy.stats.rayleigh.fit(imgs_1_masked, loc=0.)
        params_2 = scipy.stats.rayleigh.fit(imgs_2_masked, loc=0.)
        print('RAYLEIGH FIT REAL:', params_1)
        print('RAYLEIGH_FIT GEN:', params_2)
        """

        class_histogram_1 = np.histogram(imgs_1_masked, bins)[0].astype(np.float32)
        class_histogram_2 = np.histogram(imgs_2_masked, bins)[0].astype(np.float32)

        class_histogram_1 /= np.sum(class_histogram_1)
        class_histogram_2 /= np.sum(class_histogram_2)

        """
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.bar(x=bins[:-1], height=class_histogram_1)
        plt.subplot(1, 2, 2)
        plt.bar(x=bins[:-1], height=class_histogram_2)
        plt.show()
        """

        js_divs[i] = jensenshannon(class_histogram_1, class_histogram_2)

        if not get_distance:
            js_divs[i] = js_divs[i] ** 2

    return js_divs


############################
### SEGMENTATION METRICS ###
############################

class Dice_Coefficient(nn.Module):

    def __init__(self, epsilon=1e-7):
        """
        Does not calculate mean over batch dimension! Returns result for every example in the batch independently.
        This is not appropriate when performing statistical tests later because this approach decreases the variance when means of batch means are taken!
        """
        super().__init__()
        self.epsilon = torch.tensor(epsilon, device=torch.device('cpu'))

    def __str__(self):
        return 'DICE'

    def forward(self, input, target):
        # shape for both, input and target: [n_batch, num_classes, h, w]
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        batch_size, num_classes, _, _ = input.shape

        input = torch.argmax(input, dim=1)  # no softmax needed
        input = one_hot(input, num_classes).permute(0, 3, 1, 2)

        dice = torch.zeros(size=(batch_size, num_classes + 2), dtype=torch.float32)

        nominators = (input * target).sum(dim=(-2, -1)) + self.epsilon
        denominators = (input + target).sum(dim=(-2, -1)) + self.epsilon

        dice[:, :num_classes] = (2.0 * nominators / denominators)
        dice[:, -2] = torch.mean(dice[:, :-2], dim=1)
        dice[:, -1] = torch.mean(dice[:, 0:-3], dim=1)

        return dice * 100


class All_Hausdorff_Distances_Chunked(nn.Module):

    def __init__(self, PARAMS):
        """
        Does not calculate mean over batch dimension! Returns result for every example in the batch independently.
        This is not appropriate when performing statistical tests later because this approach decreases the variance when means of batch means are taken!
        """
        super().__init__()
        self.percentiles = PARAMS['TRUNCATED_HAUSDORFF_PERCENTILES']
        self.ignore_class_index = PARAMS['HD_IGNORE_CLASS_INDEX']
        self.ignore_failed_hd = PARAMS['IGNORE_FAILED_HD']

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):

        batch_size, num_classes, h, w = A_data.shape

        MHD = np.zeros(shape=(batch_size, 2 + len(self.percentiles), num_classes + 2), dtype=np.float32)
        FHD = np.zeros(shape=(batch_size, 2 + len(self.percentiles), num_classes + 2), dtype=np.float32)
        RHD = np.zeros(shape=(batch_size, 2 + len(self.percentiles), num_classes + 2), dtype=np.float32)
        # samples_to_delete = []
        for i in range(batch_size):
            for j in range(num_classes):
                if j == self.ignore_class_index:
                    continue
                if as_img:
                    # Get voxel coordinates with 1 entries --> shape: [n_voxels_with_1_entries, 3]
                    A = np.argwhere(A_data[i, j])
                    B = np.argwhere(B_data[i, j])
                else:
                    A = A_data
                    B = B_data

                if len(A) == 0:
                    print('No masked pixels for sample {0} and class {1}.'.format(i, j))
                    # samples_to_delete.append(i)
                    if self.ignore_failed_hd:
                        FHD[i, :, j] = np.nan
                        RHD[i, :, j] = np.nan
                        MHD[i, :, j] = np.nan
                    else:
                        FHD[i, :, j] = np.array((h + w) / 4)
                        RHD[i, :, j] = np.array((h + w) / 4)
                        MHD[i, :, j] = np.array((h + w) / 4)
                    
                    # _ = plotting.plot_img_pred_overlay(A_data[i, j, ...], B_data[i, j, ...])
                    # plt.savefig('HD_error_sample_{0}_class{1}.png'.format(i, j), dpi=200)
                    # plt.close()
                    continue

                # D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))

                forward = np.full(shape=(A.shape[0]), fill_value=np.inf, dtype=np.float32)
                reverse = np.full(shape=(B.shape[0]), fill_value=np.inf, dtype=np.float32)
                # D_mat = cdist(A, B)
                chunk_generator = sklearn.metrics.pairwise_distances_chunked(A, B, n_jobs=1)

                forward_counter = 0
                for chunk in chunk_generator:
                    forward_chunk = np.min(chunk, axis=1)
                    reverse_chunk = np.min(chunk, axis=0)

                    reverse[reverse > reverse_chunk] = reverse_chunk[reverse_chunk < reverse]
                    forward[forward_counter:forward_counter + len(forward_chunk)] = forward_chunk
                    forward_counter += len(forward_chunk)

                # Real Hausdorff distance
                forward_max = np.max(forward)
                reverse_max = np.max(reverse)
                
                FHD[i, 0, j] = forward_max
                RHD[i, 0, j] = reverse_max
                MHD[i, 0, j] = np.max(np.array([forward_max, reverse_max]))

                # Modified Hausdorff distance
                forward_mean = np.mean(forward)
                reverse_mean = np.mean(reverse)
                FHD[i, 1, j] = forward_mean
                RHD[i, 1, j] = reverse_mean
                MHD[i, 1, j] = np.max(np.array([forward_mean, reverse_mean]))

                # Truncated Hausdorff distances
                forward_percentiles = np.zeros(shape=(len(self.percentiles, )))
                reverse_percentiles = np.zeros(shape=(len(self.percentiles, )))
                for k in range(len(self.percentiles)):
                    forward_percentiles[k] = np.percentile(forward, self.percentiles[k])
                    reverse_percentiles[k] = np.percentile(reverse, self.percentiles[k])
                FHD[i, 2:, j] = forward_percentiles
                RHD[i, 2:, j] = reverse_percentiles
                MHD[i, 2:, j] = np.max(np.array([forward_percentiles, reverse_percentiles]), axis=0)

        """
        if samples_to_delete:
            FHD = np.delete(FHD, samples_to_delete, axis=0)
            RHD = np.delete(RHD, samples_to_delete, axis=0)
            MHD = np.delete(MHD, samples_to_delete, axis=0)
        """

        FHD[:, :, -2] = np.mean(FHD[:, :, :-2], axis=-1)
        FHD[:, :, -1] = np.mean(FHD[:, :, 1:-2], axis=-1)
        RHD[:, :, -2] = np.mean(RHD[:, :, :-2], axis=-1)
        RHD[:, :, -1] = np.mean(RHD[:, :, 1:-2], axis=-1)
        MHD[:, :, -2] = np.mean(MHD[:, :, :-2], axis=-1)
        MHD[:, :, -1] = np.mean(MHD[:, :, 1:-2], axis=-1)

        return MHD, FHD, RHD

    def forward(self, predictions, labels):
        # [n_batch, c, h, w]
        # predictions = F.softmax(predictions, dim=1) does not change the result of argmax
        num_classes = predictions.shape[1]
        predictions = torch.argmax(predictions, dim=1, keepdim=False)
        # predictions = make_one_hot(predictions, C=self.num_classes)
        predictions = one_hot(predictions, num_classes).permute(0, 3, 1, 2)
        predictions = predictions.numpy()

        labels = labels.numpy()

        return self.Hausdorff_Dist(predictions, labels)


class Hausdorff_Distance(nn.Module):
    def __init__(self, PARAMS):
        """
        Faster than All_Hausdorff_Distances_Chunked, because just ordinary HD is calculated with efficient algorithm.
        Does not calculate mean over batch dimension! Returns result for every example in the batch independently.
        This is not appropriate when performing statistical tests later because this approach decreases the variance when means of batch means are taken!
        """
        super().__init__()

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):
        batch_size, num_classes, _, _ = A_data.shape
        HD = np.zeros(shape=(batch_size, num_classes + 2), dtype=np.float32)
        for i in range(batch_size):
            for j in range(num_classes):
                if as_img:
                    # Get voxel coordinates with 1 entries --> shape: [n_voxels_with_1_entries, 3]
                    A = np.argwhere(A_data[i, j, :, :] == 1)
                    B = np.argwhere(B_data[i, j, :, :] == 1)
                else:
                    A = A_data
                    B = B_data

                HD[i, j] = max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])
        HD[:, -2] = np.mean(HD[:, :-2], axis=-1)  # with background class
        HD[:, -1] = np.mean(HD[:, 1:-2], axis=-1)  # without background class
        return HD

    def forward(self, predictions, labels):
        # predictions = F.softmax(predictions, dim=1) does not change the result of argmax
        num_classes = predictions.shape[1]
        predictions = torch.argmax(predictions, dim=1, keepdim=False)
        predictions = one_hot(predictions, num_classes).permute(0, 3, 1, 2)
        predictions = predictions.numpy()
        labels = labels.numpy()

        return [self.Hausdorff_Dist(predictions, labels)]


class Hausdorff_Distance_Binary_Image(nn.Module):
    # A Haudorff distance optimized for binary pixel masks
    def __init__(self, PARAMS):
        super().__init__()

        self.ignore_class_index = PARAMS['HD_IGNORE_CLASS_INDEX']
        self.ignore_failed_hd = PARAMS['IGNORE_FAILED_HD']

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):

        batch_size, num_classes, h, w = A_data.shape

        HD = np.zeros(shape=(batch_size, num_classes + 2), dtype=np.float32)
        for i in range(batch_size):
            for j in range(num_classes):
                if j == self.ignore_class_index:
                    continue
                img_a = A_data[i, j]
                img_b = B_data[i, j]
                boundary_a = get_boundary(img_a) * (1 - img_b)
                boundary_b = get_boundary(img_b)
                list_a = np.argwhere(boundary_a)
                list_b = np.argwhere(boundary_b)
                if len(list_a) > 0 and len(list_b) > 0:
                    D_mat = cdist(list_a, list_b)
                    min_distances = np.min(D_mat, axis=1)
                    HD[i, j] = np.max(min_distances)
                else:
                    if self.ignore_failed_hd:
                        HD[i, j] = np.nan
                    else:
                        HD[i, j] = np.array((h + w) / 4)
        HD[:, -2] = np.mean(HD[:, :-2], axis=1)
        HD[:, -1] = np.mean(HD[:, :-3], axis=1)
        return HD

    def forward(self, predictions, labels):
        # predictions = F.softmax(predictions, dim=1)
        num_classes = predictions.shape[1]
        predictions = torch.argmax(predictions, dim=1, keepdim=True)
        predictions = make_one_hot(predictions, C=num_classes)
        predictions = predictions.numpy()

        labels = labels.numpy()

        return self.Hausdorff_Dist(predictions, labels)


class Modified_Hausdorff_Distance_Binary_Image(nn.Module):
    def __init__(self, PARAMS):
        super().__init__()

        self.ignore_class_index = PARAMS['HD_IGNORE_CLASS_INDEX']
        self.ignore_failed_hd = PARAMS['IGNORE_FAILED_HD']

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):

        batch_size, num_classes, h, w = A_data.shape

        failed_hd_counter = np.zeros(shape=(num_classes + 2, ))

        HD = np.zeros(shape=(batch_size, num_classes + 2), dtype=np.float32)
        for i in range(batch_size):
            for j in range(num_classes):
                if j == self.ignore_class_index:
                    continue
                img_a = A_data[i, j]  # predictions!
                img_b = B_data[i, j]

                if not np.any(img_a):
                    failed_hd_counter[j] += 1
                    if self.ignore_failed_hd:
                        HD[i, j] = np.nan
                    else:
                        HD[i, j] = np.array((h + w) / 4)
                    continue

                list_a_forward = np.argwhere(img_a * (1 - img_b))  # Problem if img_a is zero everywhere although img_b is not.
                list_b_forward = np.argwhere(get_boundary(img_b))

                list_b_backward = np.argwhere(img_b * (1 - img_a))
                list_a_backward = np.argwhere(get_boundary(img_a))

                if len(list_a_forward) > 0 and len(list_b_forward) > 0:  # list_a has zero length if mask_a lies completely in mask_b
                    D_mat = cdist(list_a_forward, list_b_forward)
                    min_distances_forward = np.min(D_mat, axis=1)
                    hd_ij_forward = np.mean(min_distances_forward) * len(list_a_forward) / len(np.argwhere(img_a))  # correction for mean value due to ignoring subset of A
                else:
                    hd_ij_forward = 0.

                if len(list_a_backward) > 0 and len(list_b_backward) > 0:
                    D_mat = cdist(list_b_backward, list_a_backward)
                    min_distances_backward = np.min(D_mat, axis=1)
                    hd_ij_backward = np.mean(min_distances_backward) * len(list_b_backward) / len(np.argwhere(img_b))
                else:
                    hd_ij_backward = 0.

                HD[i, j] = np.max([hd_ij_forward, hd_ij_backward])

        HD[:, -2] = np.mean(HD[:, :-2], axis=1)
        HD[:, -1] = np.mean(HD[:, :-3], axis=1)

        failed_hd_counter[-2] = np.mean(failed_hd_counter[:-2])
        failed_hd_counter[-1] = np.mean(failed_hd_counter[1:-2])

        return HD, failed_hd_counter

    def forward(self, predictions, labels):
        # predictions = F.softmax(predictions, dim=1)
        num_classes = predictions.shape[1]
        predictions = torch.argmax(predictions, dim=1, keepdim=True)
        predictions = make_one_hot(predictions, C=num_classes)
        predictions = predictions.numpy()

        labels = labels.numpy()

        return self.Hausdorff_Dist(predictions, labels)


class All_Hausdorff_Distances_Binary_Image(nn.Module):
    def __init__(self, PARAMS):
        super().__init__()

        self.HD = Hausdorff_Distance_Binary_Image(PARAMS)
        self.mod_HD = Modified_Hausdorff_Distance_Binary_Image(PARAMS)

    def forward(self, masks_pred, masks_true):
        num_classes = masks_pred.shape[1]
        masks_pred = torch.argmax(masks_pred, dim=1)
        masks_pred = one_hot(masks_pred, num_classes).permute(0, 3, 1, 2)
        masks_pred = masks_pred.numpy()

        masks_true = masks_true.numpy()

        HD = self.HD.Hausdorff_Dist(masks_pred, masks_true)
        mod_HD, failed_hd_counter = self.mod_HD.Hausdorff_Dist(masks_pred, masks_true)

        total_HD = np.stack([HD, mod_HD], axis=1)   

        #return [total_HD], failed_hd_counter
        return [total_HD]

##################
### DEPRECATED ###
##################

class Modified_Hausdorff_Distance(nn.Module):
    def __init__(self, PARAMS):
        super().__init__()
        try:
            self.num_classes = PARAMS['NUM_CLASSES']
        except KeyError:
            self.num_classes = PARAMS['NUM_CLASSES_MASKS']
        self.ignore_class_index = PARAMS['HD_IGNORE_CLASS_INDEX']

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):

        FHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        RHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        MHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        batch_size = A_data.shape[0]
        batch_counter = 0
        for i in range(batch_size):
            for j in range(self.num_classes):
                if j == self.ignore_class_index:
                    continue
                if as_img:
                    # Get voxel coordinates with 1 entries --> shape: [n_voxels_with_1_entries, 3]
                    A = np.argwhere(A_data[i, j, :, :] == 1)
                    B = np.argwhere(B_data[i, j, :, :] == 1)
                else:
                    A = A_data
                    B = B_data
                # D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
                D_mat = cdist(A, B)
                forward = np.mean(np.min(D_mat, axis=1))
                FHD[j] += forward

                reverse = np.mean(np.min(D_mat, axis=0))
                RHD[j] += reverse

                MHD[j] += np.max(np.array([forward, reverse]))

            batch_counter += 1
        FHD[:-2] /= batch_counter
        FHD[-2] = np.mean(FHD[:-2])
        FHD[-1] = np.mean(FHD[1:-2])
        RHD[:-2] /= batch_counter
        RHD[-2] = np.mean(RHD[:-2])
        RHD[-1] = np.mean(RHD[1:-2])
        MHD[:-2] /= batch_counter
        MHD[-2] = np.mean(MHD[:-2])
        MHD[-1] = np.mean(MHD[1:-2])
        # print('TOTAL TIME FOR HD', (time.time()-t0)/60, 'min')
        return MHD, FHD, RHD

    def forward(self, predictions, labels):
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1, keepdim=True)
        predictions = make_one_hot(predictions, C=self.num_classes)
        predictions = predictions.numpy()

        labels = labels.numpy()

        return self.Hausdorff_Dist(predictions, labels)


class Modified_Hausdorff_Distance_Pytorch(nn.Module):


    def __init__(self, PARAMS):
        super().__init__()
        try:
            self.num_classes = PARAMS['NUM_CLASSES']
        except KeyError:
            self.num_classes = PARAMS['NUM_CLASSES_MASKS']
        self.ignore_class_index = PARAMS['HD_IGNORE_CLASS_INDEX']

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):

        FHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        RHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        MHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        batch_size = A_data.shape[0]
        batch_counter = 0
        t0 = time.time()
        for i in range(batch_size):
            for j in range(self.num_classes):
                if j == self.ignore_class_index:
                    continue
                if as_img:
                    # Get voxel coordinates with 1 entries --> shape: [n_voxels_with_1_entries, 3]
                    A = torch.nonzero(A_data[i, j, :, :]).type(
                        torch.int32).detach()
                    B = torch.nonzero(B_data[i, j, :, :]).type(
                        torch.int32).detach()
                else:
                    A = A_data
                    B = B_data
                #D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
                D_mat = cdist_pytorch(A, B)
                if D_mat.shape[0] == 0:
                    FHD[j] += np.inf
                    RHD[j] += np.inf
                    MHD[j] += np.inf
                    continue

                forward = D_mat.min(dim=1)[0].mean().cpu()
                FHD[j] += forward.numpy()

                reverse = D_mat.min(dim=0)[0].mean().cpu()
                RHD[j] += reverse.numpy()

                MHD[j] += torch.tensor([forward, reverse],
                                       requires_grad=False).max().numpy()

            batch_counter += 1
        FHD[:-2] /= batch_counter
        FHD[-2] = np.mean(FHD[:-2])
        FHD[-1] = np.mean(FHD[1:-2])
        RHD[:-2] /= batch_counter
        RHD[-2] = np.mean(RHD[:-2])
        RHD[-1] = np.mean(RHD[1:-2])
        MHD[:-2] /= batch_counter
        MHD[-2] = np.mean(MHD[:-2])
        MHD[-1] = np.mean(MHD[1:-2])
        #print('TOTAL TIME FOR HD', (time.time()-t0)/60, 'min')
        return MHD, FHD, RHD

    def forward(self, predictions, labels):
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1, keepdim=True)
        predictions = make_one_hot(predictions, C=self.num_classes)

        return self.Hausdorff_Dist(predictions, labels)


class Truncated_Hausdorff_Distance(nn.Module):
    def __init__(self, PARAMS):
        super().__init__()
        try:
            self.num_classes = PARAMS['NUM_CLASSES']
        except KeyError:
            self.num_classes = PARAMS['NUM_CLASSES_MASKS']
        self.percentile = PARAMS['TRUNCATED_HAUSDORFF_PERCENTILE']
        self.ignore_class_index = PARAMS['HD_IGNORE_CLASS_INDEX']

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):

        FHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        RHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        MHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        batch_size = A_data.shape[0]
        batch_counter = 0
        t0 = time.time()
        for i in range(batch_size):
            for j in range(self.num_classes):
                if j == self.ignore_class_index:
                    continue
                if as_img:
                    # Get voxel coordinates with 1 entries --> shape: [n_voxels_with_1_entries, 3]
                    A = np.argwhere(A_data[i, j, :, :] == 1)
                    B = np.argwhere(B_data[i, j, :, :] == 1)
                else:
                    A = A_data
                    B = B_data
                #D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
                D_mat = cdist(A, B)
                forward = np.min(D_mat, axis=1)
                forward = np.percentile(forward, self.percentile)
                reverse = np.min(D_mat, axis=0)
                reverse = np.percentile(reverse, self.percentile)

                FHD[j] += forward
                RHD[j] += reverse
                MHD[j] += np.max(np.array([forward, reverse]))

            batch_counter += 1
        FHD[:-2] /= batch_counter
        FHD[-2] = np.mean(FHD[:-2])
        FHD[-1] = np.mean(FHD[1:-2])
        RHD[:-2] /= batch_counter
        RHD[-2] = np.mean(RHD[:-2])
        RHD[-1] = np.mean(RHD[1:-2])
        MHD[:-2] /= batch_counter
        MHD[-2] = np.mean(MHD[:-2])
        MHD[-1] = np.mean(MHD[1:-2])
        #print('TOTAL TIME FOR HD', (time.time()-t0)/60, 'min')
        return MHD, FHD, RHD

    def forward(self, predictions, labels):
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1, keepdim=True)
        predictions = make_one_hot(predictions, C=self.num_classes)
        predictions = predictions.numpy()

        labels = labels.numpy()

        return self.Hausdorff_Dist(predictions, labels)


class Truncated_Hausdorff_Distance_Pytorch(nn.Module):
    def __init__(self, PARAMS):
        super().__init__()
        try:
            self.num_classes = PARAMS['NUM_CLASSES']
        except KeyError:
            self.num_classes = PARAMS['NUM_CLASSES_MASKS']
        self.percentile = PARAMS['TRUNCATED_HAUSDORFF_PERCENTILE']
        self.ignore_class_index = PARAMS['HD_IGNORE_CLASS_INDEX']

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):

        FHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        RHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        MHD = np.zeros(shape=(self.num_classes + 2, ), dtype=np.float32)
        batch_size = A_data.shape[0]
        batch_counter = 0
        t0 = time.time()
        for i in range(batch_size):
            for j in range(self.num_classes):
                if j == self.ignore_class_index:
                    continue
                if as_img:
                    # Get voxel coordinates with 1 entries --> shape: [n_voxels_with_1_entries, 3]
                    A = torch.nonzero(A_data[i, j, :, :])
                    B = torch.nonzero(B_data[i, j, :, :])
                else:
                    A = A_data
                    B = B_data
                #D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
                D_mat = cdist_pytorch(A, B)
                if D_mat.shape[0] == 0:
                    FHD[j] += np.inf
                    RHD[j] += np.inf
                    MHD[j] += np.inf
                    continue
                forward = D_mat.min(dim=1)[0].numpy()
                forward = np.percentile(forward, self.percentile)
                reverse = D_mat.min(dim=0)[0].numpy()
                reverse = np.percentile(reverse, self.percentile)

                FHD[j] += forward
                RHD[j] += reverse
                MHD[j] += np.max(np.array([forward, reverse]))

            batch_counter += 1
        FHD[:-2] /= batch_counter
        FHD[-2] = np.mean(FHD[:-2])
        FHD[-1] = np.mean(FHD[1:-2])
        RHD[:-2] /= batch_counter
        RHD[-2] = np.mean(RHD[:-2])
        RHD[-1] = np.mean(RHD[1:-2])
        MHD[:-2] /= batch_counter
        MHD[-2] = np.mean(MHD[:-2])
        MHD[-1] = np.mean(MHD[1:-2])
        #print('TOTAL TIME FOR HD', (time.time()-t0)/60, 'min')
        return MHD, FHD, RHD

    def forward(self, predictions, labels):
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1, keepdim=True)
        predictions = make_one_hot(predictions, C=self.num_classes)

        return self.Hausdorff_Dist(predictions, labels)


class All_Hausdorff_Distances(nn.Module):
    def __init__(self, PARAMS):
        super().__init__()
        try:
            self.num_classes = PARAMS['NUM_CLASSES']
        except KeyError:
            self.num_classes = PARAMS['NUM_CLASSES_MASKS']
        self.percentiles = PARAMS['TRUNCATED_HAUSDORFF_PERCENTILES']
        self.ignore_class_index = PARAMS['HD_IGNORE_CLASS_INDEX']

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):

        MHD = np.zeros(shape=(2 + len(self.percentiles),
                              self.num_classes + 2, ), dtype=np.float32)
        FHD = np.zeros(shape=(2 + len(self.percentiles),
                              self.num_classes + 2,), dtype=np.float32)
        RHD = np.zeros(shape=(2 + len(self.percentiles),
                              self.num_classes + 2,), dtype=np.float32)
        batch_size = A_data.shape[0]
        batch_counter = 0
        t0 = time.time()
        for i in range(batch_size):
            for j in range(self.num_classes):
                if j == self.ignore_class_index:
                    continue
                if as_img:
                    # Get voxel coordinates with 1 entries --> shape: [n_voxels_with_1_entries, 3]
                    A = np.argwhere(A_data[i, j, :, :] == 1)
                    B = np.argwhere(B_data[i, j, :, :] == 1)
                else:
                    A = A_data
                    B = B_data
                #D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
                if A.shape[0] == 0 or B.shape[0] == 0:
                    batch_counter -= 1
                    break
                D_mat = cdist(A, B)
                forward = np.min(D_mat, axis=1)
                reverse = np.min(D_mat, axis=0)
                del D_mat

                # Real Hausdorff distance
                forward_max = np.max(forward)
                reverse_max = np.max(reverse)
                FHD[0, j] += forward_max
                RHD[0, j] += reverse_max
                MHD[0, j] += np.max(np.array([forward_max, reverse_max]))

                # Modified Hausdorff distance
                forward_mean = np.mean(forward)
                reverse_mean = np.mean(reverse)
                FHD[1, j] += forward_mean
                RHD[1, j] += reverse_mean
                MHD[1, j] += np.max(np.array([forward_mean, reverse_mean]))

                # Truncated Hausdorff distances
                forward_percentiles = np.zeros(shape=(len(self.percentiles, )))
                reverse_percentiles = np.zeros(shape=(len(self.percentiles, )))
                for k in range(len(self.percentiles)):
                    forward_percentiles[k] = np.percentile(
                        forward, self.percentiles[k])
                    reverse_percentiles[k] = np.percentile(
                        reverse, self.percentiles[k])
                FHD[2:, j] += forward_percentiles
                FHD[2:, j] += reverse_percentiles
                MHD[2:, j] += np.max(np.array([forward_percentiles,
                                               reverse_percentiles]), axis=0)

            batch_counter += 1
        FHD[:, :-2] /= batch_counter
        FHD[:, -2] = np.mean(FHD[:, :-2], axis=1)
        FHD[:, -1] = np.mean(FHD[:, 1:-2], axis=1)
        RHD[:, :-2] /= batch_counter
        RHD[:, -2] = np.mean(RHD[:, :-2], axis=1)
        RHD[:, -1] = np.mean(RHD[:, 1:-2], axis=1)
        MHD[:, :-2] /= batch_counter
        MHD[:, -2] = np.mean(MHD[:, :-2], axis=1)
        MHD[:, -1] = np.mean(MHD[:, 1:-2], axis=1)
        #print('TOTAL TIME FOR HD', (time.time()-t0)/60, 'min')
        return MHD, FHD, RHD

    def forward(self, predictions, labels):
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1, keepdim=True)
        predictions = make_one_hot(predictions, C=self.num_classes)
        predictions = predictions.numpy()

        labels = labels.numpy()

        return self.Hausdorff_Dist(predictions, labels)


class All_Hausdorff_Distances_Split(nn.Module):
    def __init__(self, PARAMS):
        super().__init__()
        try:
            self.num_classes = PARAMS['NUM_CLASSES']
        except KeyError:
            self.num_classes = PARAMS['NUM_CLASSES_MASKS']
        self.percentiles = PARAMS['TRUNCATED_HAUSDORFF_PERCENTILES']
        self.ignore_class_index = PARAMS['HD_IGNORE_CLASS_INDEX']

    def Hausdorff_Dist(self, A_data, B_data, as_img=True):

        MHD = np.zeros(shape=(2 + len(self.percentiles),
                              self.num_classes + 2, ), dtype=np.float32)
        FHD = np.zeros(shape=(2 + len(self.percentiles),
                              self.num_classes + 2,), dtype=np.float32)
        RHD = np.zeros(shape=(2 + len(self.percentiles),
                              self.num_classes + 2,), dtype=np.float32)
        batch_size = A_data.shape[0]
        batch_counter = 0
        t0 = time.time()
        for i in range(batch_size):
            for j in range(self.num_classes):
                if j == self.ignore_class_index:
                    continue
                if as_img:
                    # Get voxel coordinates with 1 entries --> shape: [n_voxels_with_1_entries, 3]
                    A = np.argwhere(A_data[i, j, :, :] == 1)
                    B = np.argwhere(B_data[i, j, :, :] == 1)
                else:
                    A = A_data
                    B = B_data
                #D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
                if A.shape[0] == 0 or B.shape[0] == 0:
                    batch_counter -= 1
                    break
                #D_mat = cdist(A, B)
                _, forward = sklearn.metrics.pairwise_distances_argmin_min(
                    A, B)
                _, reverse = sklearn.metrics.pairwise_distances_argmin_min(
                    B, A)

                print(forward.shape)
                print(reverse.shape)

                #forward = np.min(D_mat, axis=1)
                #reverse = np.min(D_mat, axis=0)
                #del D_mat

                # Real Hausdorff distance
                forward_max = np.max(forward)
                reverse_max = np.max(reverse)
                FHD[0, j] += forward_max
                RHD[0, j] += reverse_max
                MHD[0, j] += np.max(np.array([forward_max, reverse_max]))

                # Modified Hausdorff distance
                forward_mean = np.mean(forward)
                reverse_mean = np.mean(reverse)
                FHD[1, j] += forward_mean
                RHD[1, j] += reverse_max
                MHD[1, j] += np.max(np.array([forward_mean, reverse_mean]))

                # Truncated Hausdorff distances
                forward_percentiles = np.zeros(shape=(len(self.percentiles, )))
                reverse_percentiles = np.zeros(shape=(len(self.percentiles, )))
                for k in range(len(self.percentiles)):
                    forward_percentiles[k] = np.percentile(
                        forward, self.percentiles[k])
                    reverse_percentiles[k] = np.percentile(
                        reverse, self.percentiles[k])
                FHD[2:, j] += forward_percentiles
                FHD[2:, j] += reverse_percentiles
                MHD[2:, j] += np.max(np.array([forward_percentiles,
                                               reverse_percentiles]), axis=0)

            batch_counter += 1
        FHD[:, :-2] /= batch_counter
        FHD[:, -2] = np.mean(FHD[:, :-2], axis=1)
        FHD[:, -1] = np.mean(FHD[:, 1:-2], axis=1)
        RHD[:, :-2] /= batch_counter
        RHD[:, -2] = np.mean(RHD[:, :-2], axis=1)
        RHD[:, -1] = np.mean(RHD[:, 1:-2], axis=1)
        MHD[:, :-2] /= batch_counter
        MHD[:, -2] = np.mean(MHD[:, :-2], axis=1)
        MHD[:, -1] = np.mean(MHD[:, 1:-2], axis=1)
        return MHD, FHD, RHD

    def forward(self, predictions, labels):
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1, keepdim=True)
        predictions = make_one_hot(predictions, C=self.num_classes)
        predictions = predictions.numpy()

        labels = labels.numpy()

        return self.Hausdorff_Dist(predictions, labels)
