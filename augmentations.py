import numpy as np
import torch


def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b


def DataTransform(sample, params):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, params["jitter_scale_ratio"])
    # weak_aug = permutation(sample, max_segments=params["max_seg"])
    strong_aug = jitter(permutation(sample, max_segments=params["max_seg"]), params["jitter_ratio"])

    return weak_aug, strong_aug

# def DataTransform_TD(sample, params):
#     """Weak and strong augmentations"""
#     weak_aug = sample
#     strong_aug = jitter(permutation(sample, max_segments=params["max_seg"]), params["jitter_ratio"]) #masking(sample)
#     return weak_aug, strong_aug
#
# def DataTransform_FD(sample, params):
#     """Weak and strong augmentations in Frequency domain """
#     # weak_aug =  remove_frequency(sample, 0.1)
#     strong_aug = add_frequency(sample, 0.1)
#     return weak_aug, strong_aug


def DataTransform_TD(sample, params):
    """Weak and strong augmentations"""
    aug_1 = jitter(sample, params["jitter_ratio"])
    aug_2 = scaling(sample, params["jitter_scale_ratio"])
    aug_3 = permutation(sample, max_segments=params["max_seg"])

    # there are three augmentations in Time domain
    li = np.random.randint(0, 3, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)
    # the rows are not selected are set as zero.
    # ll = li_onehot[:, 0]
    # x = 1-li_onehot[:, 0]

    aug_1[1 - li_onehot[:, 0]] = 0
    aug_2[1 - li_onehot[:, 1]] = 0
    aug_3[1 - li_onehot[:, 2]] = 0
    # aug_4[1 - li_onehot[:, 3]] = 0
    aug_T = aug_1 + aug_2 + aug_3 #+aug_4
    return aug_T


def DataTransform_FD(sample, params):
    """Weak and strong augmentations in Frequency domain """
    # there are two augmentations in Frequency domain
    aug_1 = remove_frequency(sample, 0.1)
    aug_2 = add_frequency(sample, 0.1)
    # generate random sequence
    li = np.random.randint(0, 2, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)
    # the rows are not selected are set as zero.
    # 一次性的针对相关的
    aug_1[1-li_onehot[:, 0]] = 0
    aug_2[1 - li_onehot[:, 1]] = 0
    aug_F = aug_1 + aug_2
    return aug_F


def generate_binomial_mask(B, T, D, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)


def masking(x, mask='binomial'):
    nan_mask = ~x.isnan().any(axis=-1)
    x[~nan_mask] = 0
    # x = self.input_fc(x)  # B x T x Ch

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=0.9).to(x.device)
    # elif mask == 'continuous':
    #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    # elif mask == 'all_true':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    # elif mask == 'all_false':
    #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    # elif mask == 'mask_last':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     mask[:, -1] = False

    # mask &= nan_mask
    x[~mask_id] = 0
    return x


# 抖动
def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


# 尺度变换
def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return torch.from_numpy(np.concatenate((ai), axis=1))


# 拼接
def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


# 去除频域
def remove_frequency(x, maskout_ratio=0):
    # maskout_ratio are False
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > maskout_ratio
    mask = mask.to(x.device)
    return x*mask


# 增强频域
def add_frequency(x, pertub_ratio=0,):
    # only pertub_ratio of all values are True
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio)
    mask = mask.to(x.device)
    max_amplitude = x.max()
    x = torch.rand(mask.shape)
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x + pertub_matrix