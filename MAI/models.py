import torch.nn as nn

from tqdm import tqdm
from einops import rearrange
from modules.utils import *
from modules.generation import *
from modules.visualization import *
from metrics.timegan_metrics import calculate_pred_disc


def mask_it(x, masks):
    # x(bs, ts_size, z_dim)
    b, l, f = x.shape
    x_visible = x[~masks, :].reshape(b, -1, f)  # (bs, vis_size, z_dim)
    return x_visible


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.z_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.hidden_dim)

    def forward(self, x):
        x_enc, _ = self.rnn(x)
        x_enc = self.fc(x_enc)
        return x_enc


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.hidden_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.z_dim)

    def forward(self, x_enc):
        x_dec, _ = self.rnn(x_enc)
        x_dec = self.fc(x_dec)
        return x_dec


class Interpolator(nn.Module):
    def __init__(self, args):
        super(Interpolator, self).__init__()
        self.sequence_inter = nn.Linear(in_features=(args.ts_size - args.total_mask_size),
                                        out_features=args.ts_size)
        self.feature_inter = nn.Linear(in_features=args.hidden_dim,
                                       out_features=args.hidden_dim)

    def forward(self, x):
        # x(bs, vis_size, hidden_dim)
        x = rearrange(x, 'b l f -> b f l')  # x(bs, hidden_dim, vis_size)
        x = self.sequence_inter(x)  # x(bs, hidden_dim, ts_size)
        x = rearrange(x, 'b f l -> b l f')  # x(bs, ts_size, hidden_dim)
        x = self.feature_inter(x)  # x(bs, ts_size, hidden_dim)
        return x


class InterpoMAEUnit(nn.Module):
    def __init__(self, args):
        super(InterpoMAEUnit, self).__init__()
        self.args = args
        self.ts_size = args.ts_size
        self.mask_size = args.mask_size
        self.num_masks = args.num_masks
        self.total_mask_size = args.num_masks * args.mask_size
        args.total_mask_size = self.total_mask_size
        self.z_dim = args.z_dim
        self.encoder = Encoder(args)
        self.interpolator = Interpolator(args)
        self.decoder = Decoder(args)

    def forward_mae(self, x, masks):
        """No mask tokens, using Interpolation in the latent space"""
        x_vis = mask_it(x, masks)  # (bs, vis_size, z_dim)
        x_enc = self.encoder(x_vis)  # (bs, vis_size, hidden_dim)
        x_inter = self.interpolator(x_enc)  # (bs, ts_size, hidden_dim)
        x_dec = self.decoder(x_inter)  # (bs, ts_size, z_dim)
        return x_inter, x_dec, masks

    def forward_ae(self, x, masks):
        """mae_pseudo_mask is equivalent to the Autoencoder
            There is no interpolator in this mode"""
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_enc, x_dec, masks

    def forward(self, x, masks, mode):
        """Existing mode:
            1. train_ae
            2. train_mae
            3. random_generation
            4. cross_generation"""
        if mode == 'train_ae':
            x_encoded, x_decoded, masks = self.forward_ae(x, masks)
        else:
            x_encoded, x_decoded, masks = self.forward_mae(x, masks)
        return x_encoded, x_decoded, masks


class InterpoMAE(nn.Module):
    def __init__(self, args, ori_data):
        super(InterpoMAE, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.model = InterpoMAEUnit(args).to(self.device)
        self.ori_data = ori_data
        self.pseudo_masks = generate_pseudo_masks(args, args.batch_size)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.num_iteration = 0
        print(f'Successfully initialized {self.__class__.__name__}!')

    def train_ae(self):
        self.model.train()

        for t in tqdm(range(self.args.ae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            x_enc, x_dec, masks = self.model(x_ori, self.pseudo_masks, 'train_ae')
            loss = self.criterion(x_dec, x_ori)

            self.num_iteration += 1

            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} total loss')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_args(self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_embed(self):
        for t in tqdm(range(self.args.embed_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            random_masks = generate_random_masks(self.args, self.args.batch_size)

            # Get the target x_ori_enc by Autoencoder
            self.model.eval()
            x_ori_enc, _, masks = self.model(x_ori, self.pseudo_masks, 'train_ae')
            x_ori_enc = x_ori_enc.clone().detach()  # (bs, ts_size, hidden_dim)
            b, l, f = x_ori_enc.size()

            self.model.train()
            x_enc, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')

            # Only calculate loss for those being masked
            x_enc_masked = x_enc[masks, :].reshape(b, -1, f)
            x_ori_enc_masked = x_ori_enc[masks, :].reshape(b, -1, f)
            loss = self.criterion(x_enc_masked, x_ori_enc_masked)
            # By annotate lines above, we take loss on all patches
            # loss = self.criterion(x_enc, x_ori_enc)  # embed_loss

            self.num_iteration += 1

            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} loss.')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_args(self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_recon(self):
        for t in tqdm(range(self.args.recon_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            random_masks = generate_random_masks(self.args, self.args.batch_size)  # (bs, ts_size)

            self.model.train()
            _, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')
            loss = self.criterion(x_dec, x_ori)

            self.num_iteration += 1

            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} loss.')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_args(self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate_ae(self):
        """Evaluate the model as a simple Anto Encoder"""
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = full_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization

        # Visualization
        plot_time_series_no_masks(self.args)
        pca_and_tsne(self.args)

        # Calculate Predictive and Discriminative Scores
        print('Calculating Pred and Disc Scores\n')
        calculate_pred_disc(self.args)

    def evaluate_random_mae(self):
        """Evaluate the model as a Masked Auto Encoder"""
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = random_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val

        # Save Renormalized art_data
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization
        print('Synthetic Data Generation Finished.')

        # Visualization
        plot_time_series_with_masks(self.args)
        pca_and_tsne(self.args)

    def synthesize_cross_average(self):
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = cross_average_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val

        # Save Renormalized art_data
        save_dir = os.path.join(self.args.synthesis_dir, 'cross_average')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        art_data_dir = os.path.join(save_dir, 'art_data.npy')
        np.save(art_data_dir, art_data)  # save art_data after renormalization
        print('Synthetic Data Generation by Cross Average Finished.')

        # Visualization
        temp_args = self.args
        temp_args.pics_dir = save_dir
        plot_time_series_no_masks(temp_args)
        pca_and_tsne(temp_args)

    def synthesize_cross_concate(self):
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = cross_concat_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val

        # Save Renormalized art_data
        save_dir = os.path.join(self.args.synthesis_dir, 'cross_concate')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        art_data_dir = os.path.join(save_dir, 'art_data.npy')
        np.save(art_data_dir, art_data)  # save art_data after renormalization
        print('Synthetic Data Generation by Cross Concate Finished.')

        # Visualization
        temp_args = self.args
        temp_args.pics_dir = save_dir
        plot_time_series_no_masks(temp_args)
        pca_and_tsne(temp_args)

    def synthesize_random_average(self):
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = random_average_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val

        # Save Renormalized art_data
        save_dir = os.path.join(self.args.synthesis_dir, 'random_average')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        art_data_dir = os.path.join(save_dir, 'art_data.npy')
        np.save(art_data_dir, art_data)  # save art_data after renormalization
        print('Synthetic Data Generation by Random Average Finished.')

        # Visualization
        temp_args = self.args
        temp_args.pics_dir = save_dir
        plot_time_series_no_masks(temp_args)
        pca_and_tsne(temp_args)
