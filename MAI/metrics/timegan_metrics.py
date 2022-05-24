from tqdm import tqdm
from modules.utils import *
from metrics.predictive_metrics import predictive_score_metrics
from metrics.discriminative_metrics import discriminative_score_metrics


def timegan_predictive(args, ori_data, art_data):
    pred_scores = list()
    for _ in tqdm(range(args.metric_iteration)):
        temp_pred = predictive_score_metrics(ori_data, art_data)
        pred_scores.append(temp_pred)
    pred_mean = np.mean(pred_scores)
    pred_std = np.std(pred_scores)
    return pred_mean, pred_std, pred_scores


def timegan_discriminative(args, ori_data, art_data):
    disc_scores = list()
    for _ in tqdm(range(args.metric_iteration)):
        temp_disc = discriminative_score_metrics(ori_data, art_data)
        disc_scores.append(temp_disc)
    disc_mean = np.mean(disc_scores)
    disc_std = np.std(disc_scores)
    return disc_mean, disc_std, disc_scores


def timegan_metrics(args, ori_data, art_data, metrics_dir):
    # Get the  predictive score with repeat
    print('Start Calculate Predictive Score.')
    pred_mean, pred_std, pred_scores = timegan_predictive(args, ori_data, art_data)
    print(f'Mean Predictive Score {pred_mean} with std {pred_std}!')

    # original_metrics_dir = os.path.join(args.model_dir, 'metrics_results.npy')
    # metrics_results = load_dict_npy(original_metrics_dir)[()]
    metrics_results = dict()
    metrics_results['pred_mean'] = pred_mean
    metrics_results['pred_std'] = pred_std
    metrics_results['pred_scores'] = pred_scores
    np.save(metrics_dir, metrics_results)

    # Get the discriminative score with repeat
    print('Start Calculate Discriminative Score.')
    disc_mean, disc_std, disc_scores = timegan_discriminative(args, ori_data, art_data)
    print(f'Mean Discriminative Score {disc_mean} with std {disc_std}!')

    metrics_results = load_dict_npy(metrics_dir)[()]
    metrics_results['disc_mean'] = disc_mean
    metrics_results['disc_std'] = disc_std
    metrics_results['disc_scores'] = disc_scores
    np.save(metrics_dir, metrics_results)

    print('Evaluation by TimeGAN style Metrics Finished.')


def calculate_pred_disc(args):
    # For Random Once
    print('For Random Once.')
    ori_data = np.load(args.ori_data_dir)
    ori_data, min_ori, max_ori = min_max_scalar(ori_data)
    art_data = np.load(args.art_data_dir)
    art_data, min_art, max_art = min_max_scalar(art_data)
    print('Data Loading and Normalization Finished.')
    
    metrics_dir = os.path.join(args.model_dir, 'metrics_results.npy')
    timegan_metrics(args, ori_data, art_data, metrics_dir)
    
    # For Cross Average
    print('For Cross Average.')
    ori_data = np.load(args.ori_data_dir)
    ori_data, min_ori, max_ori = min_max_scalar(ori_data)

    cross_average_dir = os.path.join(args.synthesis_dir, 'cross_average')
    cross_average_data_dir = os.path.join(cross_average_dir, 'art_data.npy')
    cross_average_data = np.load(cross_average_data_dir)
    art_data, min_art, max_art = min_max_scalar(cross_average_data)
    
    metrics_dir = os.path.join(cross_average_dir, 'metrics_results.npy')
    timegan_metrics(args, ori_data, art_data, metrics_dir)

    # For Cross Concate
    print('For Cross Concate')
    ori_data = np.load(args.ori_data_dir)
    ori_data, min_ori, max_ori = min_max_scalar(ori_data)

    cross_concate_dir = os.path.join(args.synthesis_dir, 'cross_concate')
    cross_concate_data_dir = os.path.join(cross_concate_dir, 'art_data.npy')
    cross_concate_data = np.load(cross_concate_data_dir)
    art_data, min_art, max_art = min_max_scalar(cross_concate_data)
    
    metrics_dir = os.path.join(cross_concate_dir, 'metrics_results.npy')
    timegan_metrics(args, ori_data, art_data, metrics_dir)
    
    # For Random Average
    print('For Random Average')
    ori_data = np.load(args.ori_data_dir)
    # ori_data, min_ori, max_ori = min_max_scalar(ori_data)
    ori_data, min_ori, max_ori = min_max_scalar(ori_data)

    random_average_dir = os.path.join(args.synthesis_dir, 'random_average')
    random_average_data_dir = os.path.join(random_average_dir, 'art_data.npy')
    random_average_data = np.load(random_average_data_dir)
    art_data, min_art, max_art = min_max_scalar(random_average_data)
    
    metrics_dir = os.path.join(random_average_dir, 'metrics_results.npy')
    timegan_metrics(args, ori_data, art_data, metrics_dir)


if __name__ == '__main__':
    home = os.getcwd()
    real_home = os.path.abspath(os.path.join(home, '..'))
    os.chdir(real_home)
    args = load_arguments(real_home)
    # For Original as the Synthetic
    print('For Original as the Synthetic.')
    ori_data = np.load(args.ori_data_dir)
    ori_data, min_ori, max_ori = min_max_scalar(ori_data)
    art_data = np.load(args.ori_data_dir)
    art_data, min_art, max_art = min_max_scalar(art_data)
    print('Data Loading and Normalization Finished.')

    metrics_dir = os.path.join(args.model_dir, 'ori_as_syn_metrics_results.npy')
    timegan_metrics(args, ori_data, art_data, metrics_dir)
