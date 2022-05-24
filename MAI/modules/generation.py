import numpy as np
import torch


def generate_pseudo_masks(args, num_samples):
    # xxxx
    # xxxx
    # xxxx
    masks = np.zeros((num_samples, args.ts_size), dtype=bool)
    return masks


def generate_random_masks(args, num_samples):
    # xxxo
    # oxxx
    # xxox
    num_patches = int(args.ts_size // args.mask_size)

    def single_sample_mask():
        idx = np.random.permutation(num_patches)[:args.num_masks]
        mask = np.zeros(args.ts_size, dtype=bool)
        for j in idx:
            mask[j * args.mask_size:(j + 1) * args.mask_size] = 1
        return mask

    masks_list = [single_sample_mask() for _ in range(num_samples)]
    masks = np.stack(masks_list, axis=0)  # (num_samples, ts_size)
    return masks


def generate_cross_masks(args, num_samples, idx):
    # oxxx
    # oxxx
    # oxxx
    masks = np.zeros((num_samples, args.ts_size), dtype=bool)
    masks[:, idx * args.total_mask_size:(idx + 1) * args.total_mask_size] = 1  # masks(num_samples, ts_size)
    return masks

"""Generation for the Evaluation and Synthesis
    1. full_generation: no patches masked, Auto Encoder style
    2. random_generation: randomly generate some masks
    3. random_average_generation_engine: random mask, average results
    4. cross_concate_generation_engine: cross mask, concate patches
    5. cross_average_generation_engine: cross mask, average results"""


def full_generation(args, model, ori_data):
    masks = generate_pseudo_masks(args, len(ori_data))  # (len(ori_data), seq_len)
    x_enc, art_data, masks = model(ori_data, masks, 'full_generation')
    np.save(args.masks_dir, masks)
    return art_data


def random_generation(args, model, ori_data):
    masks = generate_random_masks(args, len(ori_data))  # (len(ori_data), seq_len)
    x_enc, art_data, masks = model(ori_data, masks, 'random_generation')
    np.save(args.masks_dir, masks)
    return art_data


def random_average_generation(args, model, ori_data):
    num_gen = 10
    generations = []
    for i in range(num_gen):
        masks = generate_random_masks(args, len(ori_data))
        _, generation, masks = model(ori_data, masks, 'random_average_generation')
        generations.append(generation)
    generations = torch.stack(generations)
    art_data = torch.mean(generations, dim=0, keepdim=False)
    return art_data


def cross_concat_generation(args, model, ori_data):
    # eg. num_mask = 2 mask_size = 4 ts_size = 20
    # args.ts_size//args.mask_size = 5
    # oo|oo|o where o = 8 positions
    # num_gen = 2
    # num_rest = 1
    num_gen = int(args.ts_size//args.total_mask_size)
    split_pos = num_gen * args.total_mask_size
    generations = []
    for i in range(num_gen):
        masks = generate_cross_masks(args, len(ori_data), i)
        _, generation, masks = model(ori_data, masks, 'cross_concat_generation')
        generations.append(generation[:, i * args.total_mask_size:(i+1) * args.total_mask_size, :])
    if split_pos != args.ts_size:
        masks = np.zeros((len(ori_data), args.ts_size), dtype=bool)
        masks[:, -args.total_mask_size:] = 1
        _, generation, masks = model(ori_data, masks, 'cross_concat_generation')
        generations.append(generation[:, split_pos:, :])
    art_data = torch.cat(generations, dim=1)
    return art_data


def cross_average_generation(args, model, ori_data):
    num_gen = int(args.ts_size//args.total_mask_size)
    split_pos = num_gen * args.total_mask_size
    generations = []
    for i in range(num_gen):
        masks = generate_cross_masks(args, len(ori_data), i)
        _, generation, masks = model(ori_data, masks, 'cross_average_generation')
        generations.append(generation)
    if split_pos != args.ts_size:
        masks = np.zeros((len(ori_data), args.ts_size), dtype=bool)
        masks[:, -args.total_mask_size:] = 1
        _, generation, masks = model(ori_data, masks, 'cross_average_generation')
        generations.append(generation)

    generations = torch.stack(generations)
    art_data = torch.mean(generations, dim=0, keepdim=False)
    return art_data
