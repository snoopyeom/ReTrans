import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import my_kl_loss
from model.AnomalyTransformer import AnomalyTransformer
from model.transformer_ae import (
    AnomalyTransformerAE,
    train_model_with_replay,
)
from data_factory.data_loader import get_loader_segment
from utils.analysis_tools import (
    plot_z_bank_tsne,
    plot_z_bank_pca,
    plot_z_bank_umap,
)
import matplotlib.pyplot as plt
import warnings

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {
        'model_type': 'transformer',
        'latent_dim': 16,
        'beta': 1.0,
        'replay_size': 1000,
        'replay_horizon': None,
        'store_mu': False,
        'freeze_after': None,
        'ema_decay': None,
        'decoder_type': 'mlp',
        'anomaly_ratio': 1.0,
        'cpd_penalty': 20,
        'min_cpd_gap': 30,
        'cpd_log_interval': 20,
        'cpd_top_k': 3,
        'cpd_extra_ranges': [(0, 4000)],
    }

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.model_tag = getattr(self, 'model_tag', self.dataset)
        self.load_model = getattr(self, 'load_model', None)
        self.train_start = getattr(self, 'train_start', 0.0)
        self.train_end = getattr(self, 'train_end', 1.0)

        self.update_count = 0
        self.cpd_indices: list[int] = []

        self.train_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='train',
            dataset=self.dataset,
            train_start=self.train_start,
            train_end=self.train_end,
            return_index=True,
        )
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset,
                                              train_start=self.train_start,
                                              train_end=self.train_end)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset,
                                              train_start=self.train_start,
                                              train_end=self.train_end)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset,
                                              train_start=self.train_start,
                                              train_end=self.train_end)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        if getattr(self, 'model_type', 'transformer') == 'transformer_ae':
            self.model = AnomalyTransformerAE(
                win_size=self.win_size,
                enc_in=self.input_c,
                latent_dim=getattr(self, 'latent_dim', 16),
                replay_size=getattr(self, 'replay_size', 1000),
                replay_horizon=getattr(self, 'replay_horizon', None),
                freeze_after=getattr(self, 'freeze_after', None),
                ema_decay=getattr(self, 'ema_decay', None),
                decoder_type=getattr(self, 'decoder_type', 'mlp'),
            )
        else:
            self.model = AnomalyTransformer(
                win_size=self.win_size,
                enc_in=self.input_c,
                c_out=self.output_c,
                e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def compute_metrics(self):
        """Evaluate F1 and ROC AUC on the current model using the threshold loader."""
        self.model.eval()
        try:
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        except ImportError:
            warnings.warn("scikit-learn is required to compute metrics")
            return float('nan'), float('nan')

        # use the modern argument name to silence PyTorch deprecation warning
        criterion = nn.MSELoss(reduction="none")
        temperature = 50

        # energies on train set
        attens_energy = []
        for i, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                input_data, labels, _ = batch
            else:
                input_data, labels = batch
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u],
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                             .repeat(1, 1, 1, self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                         .repeat(1, 1, 1, self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u],
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                             .repeat(1, 1, 1, self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                         .repeat(1, 1, 1, self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
        train_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        # energies on threshold loader
        attens_energy = []
        test_labels = []
        for i, batch in enumerate(self.thre_loader):
            if len(batch) == 3:
                input_data, labels, _ = batch
            else:
                input_data, labels = batch
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u],
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                             .repeat(1, 1, 1, self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                         .repeat(1, 1, 1, self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u],
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                             .repeat(1, 1, 1, self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                         .repeat(1, 1, 1, self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        combined_energy = np.concatenate([train_energy, attens_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        pred = (attens_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        from sklearn.metrics import precision_recall_fscore_support
        # detection adjustment: please see this issue for more information
        # https://github.com/thuml/Anomaly-Transformer/issues/14
        pred = list(pred)
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        if len(np.unique(gt)) < 2:
            warnings.warn(
                "Only one class present in y_true. F1 and ROC AUC are undefined"
            )
            return float("nan"), float("nan")

        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0)
        auc = roc_auc_score(gt, attens_energy)

        return f_score, auc

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.model_tag)
        train_steps = len(self.train_loader)
        self.history = []

        if self.load_model is not None and os.path.isfile(self.load_model):
            self.model.load_state_dict(torch.load(self.load_model))
            print(f"Loaded pretrained model from {self.load_model}")

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                if len(batch) == 3:
                    input_data, labels, indices = batch
                else:
                    input_data, labels = batch
                    indices = None

                iter_count += 1
                input = input_data.float().to(self.device)

                if getattr(self, 'model_type', 'transformer') == 'transformer_ae':
                    loss, updated = train_model_with_replay(
                        self.model,
                        self.optimizer,
                        input,
                        indices=indices,
                        cpd_penalty=getattr(self, 'cpd_penalty', 20),
                        min_gap=getattr(self, 'min_cpd_gap', 30),
                    )
                    loss1_list.append(loss)
                    if updated:
                        self.update_count += 1
                        if indices is not None and len(indices) > 0:
                            self.cpd_indices.append(int(indices[0]))
                        if self.update_count % getattr(self, 'cpd_log_interval', 20) == 0:
                            # evaluate periodically after concept drift update
                            vali_loss1, vali_loss2 = self.vali(self.test_loader)
                            f1, auc = self.compute_metrics()
                            self.history.append((self.update_count, f1, auc))
                            print(
                                f"Update {self.update_count}: Val Loss {vali_loss1:.4f} "
                                f"F1 {f1:.4f} AUC {auc:.4f}"
                            )
                        else:
                            # mark the update without expensive evaluation
                            self.history.append((self.update_count, float('nan'), float('nan')))
                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    continue

                self.optimizer.zero_grad()
                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)
            need_metrics = (
                not self.history
                or self.history[-1][0] != self.update_count
                or np.isnan(self.history[-1][1])
                or np.isnan(self.history[-1][2])
            )
            if need_metrics:
                # compute metrics if none recorded for this update or if NaN
                f1, auc = self.compute_metrics()
                if not self.history or self.history[-1][0] != self.update_count:
                    self.history.append((self.update_count, f1, auc))
                else:
                    self.history[-1] = (self.update_count, f1, auc)
            else:
                f1, auc = self.history[-1][1], self.history[-1][2]

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f} F1: {4:.4f} AUC: {5:.4f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss1, f1, auc))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        if getattr(self, 'model_type', 'transformer') == 'transformer_ae':
            print(f"CPD triggered updates: {self.update_count}")
            if self.history:
                counts, f1s, aucs = zip(*self.history)
                counts = np.array(counts)
                f1s = np.array(f1s, dtype=float)
                valid_f1 = ~np.isnan(f1s)
                fig, ax = plt.subplots()
                ax.plot(counts[valid_f1], f1s[valid_f1], marker='o', linestyle='-')
                ax.set_xlabel('CPD Updates')
                ax.set_ylabel('F1 Score')
                ax.set_title('F1 Score over Updates')
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(os.path.join(path, 'f1_score.png'))
                plt.close(fig)

                counts = np.array(counts)
                aucs = np.array(aucs, dtype=float)
                valid_auc = ~np.isnan(aucs)
                fig, ax = plt.subplots()
                ax.plot(counts[valid_auc], aucs[valid_auc], marker='x', linestyle='-', color='tab:red')
                ax.set_xlabel('CPD Updates')
                ax.set_ylabel('ROC AUC')
                ax.set_title('ROC AUC over Updates')
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(os.path.join(path, 'roc_auc.png'))
                plt.close(fig)

                # additional diagnostics
                try:
                    tsne_path = os.path.join(path, 'z_bank_tsne.png')
                    plot_z_bank_tsne(self.model, self.train_loader, save_path=tsne_path)
                    pca_path = os.path.join(path, 'z_bank_pca.png')
                    plot_z_bank_pca(self.model, self.train_loader, save_path=pca_path)
                    umap_path = os.path.join(path, 'z_bank_umap.png')
                    plot_z_bank_umap(self.model, self.train_loader, save_path=umap_path)
                except Exception as e:
                    warnings.warn(f"Failed to create latent plots: {e}")

                # Visualization of detected change points over the entire
                # validation series produced a very cluttered figure. Omit the
                # global view and rely on optional zoomed-in plots instead.

    def test(self):
        ckpt_path = self.load_model
        if ckpt_path is None:
            ckpt_path = os.path.join(str(self.model_save_path), str(self.model_tag) + '_checkpoint.pth')
        # load weights only to avoid warnings on newer PyTorch versions
        self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction="none")

        # (1) stastic on the train set
        attens_energy = []
        for i, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                input_data, labels, _ = batch
            else:
                input_data, labels = batch
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, batch in enumerate(self.thre_loader):
            if len(batch) == 3:
                input_data, labels, _ = batch
            else:
                input_data, labels = batch
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, batch in enumerate(self.thre_loader):
            if len(batch) == 3:
                input_data, labels, _ = batch
            else:
                input_data, labels = batch
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score, roc_auc_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        auc = roc_auc_score(gt, test_energy)
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score, auc))

        return accuracy, precision, recall, f_score, auc
