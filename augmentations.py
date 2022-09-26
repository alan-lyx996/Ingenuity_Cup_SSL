import numpy as np
import torch


def one_hot_encoding(X):
    """将标签进行onehot编码"""
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b


def DataTransform(sample, params):
    """弱增强或强增强"""
    weak_aug = scaling(sample, params["jitter_scale_ratio"])
    strong_aug = jitter(permutation(sample, max_segments=params["max_seg"]), params["jitter_ratio"])

    return weak_aug, strong_aug


def DataTransform_TD(sample, params):
    """时域增强"""
    # 进行噪声扰动增强
    aug_1 = jitter(sample, params["jitter_ratio"])
    # 进行尺度放缩增强
    aug_2 = scaling(sample, params["jitter_scale_ratio"])
    # 进行随机置换增强
    aug_3 = permutation(sample, max_segments=params["max_seg"])

    # 时域增强是上述三种增强的合成
    # 随机生成数据增强的序列，即[0, 1, 2, ...,1, 2, 0, ...]，长度等于数据集的大小
    li = np.random.randint(0, 3, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)

    # li_onehot的维度是[length, 3],那么1-li_onehot[:,0]的维度就是[length,1],且只有0与1两个元素，
    # 即可以完成bool选择，选择出元素为0的保留，元素为1的向量置为0
    aug_1[1 - li_onehot[:, 0]] = 0
    aug_2[1 - li_onehot[:, 1]] = 0
    aug_3[1 - li_onehot[:, 2]] = 0
    # 将三种增强的结果累加在一起，构成了增强数据库
    aug_T = aug_1 + aug_2 + aug_3
    return aug_T


def DataTransform_FD(sample, params):
    """频域增强"""
    # 去除频谱增强
    aug_1 = remove_frequency(sample, 0.1)
    # 增加频谱增强
    aug_2 = add_frequency(sample, 0.1)

    # 与时域增强相似的操作
    li = np.random.randint(0, 2, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)
    aug_1[1-li_onehot[:, 0]] = 0
    aug_2[1 - li_onehot[:, 1]] = 0
    aug_F = aug_1 + aug_2
    return aug_F


def generate_binomial_mask(B, T, D, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)


def masking(x, mask='binomial'):
    """掩膜增强"""
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



def jitter(x, sigma=0.8):
    """抖动增强"""
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    """尺度变换增强"""
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return torch.from_numpy(np.concatenate((ai), axis=1))


def permutation(x, max_segments=5, seg_mode="random"):
    """合并拼接增强"""
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


def remove_frequency(x, maskout_ratio=0):
    """去除频谱增强"""
    # maskout_ratio are False
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > maskout_ratio
    mask = mask.to(x.device)
    return x*mask


def add_frequency(x, pertub_ratio=0,):
    """增大频谱增强"""
    # only pertub_ratio of all values are True
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio)
    mask = mask.to(x.device)
    max_amplitude = x.max()
    x = torch.rand(mask.shape)
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x + pertub_matrix