import argparse
import os
import sqlite3
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.covariance import EmpiricalCovariance, GraphicalLasso
from utils.outputs import GWNOutput

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
from pathlib import Path
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, auc, precision_recall_curve, roc_curve, confusion_matrix
from torch.optim import Adam, lr_scheduler
from utils.config import GWNConfig
from utils.data_provider import get_loader, rolling_window_2D, cutting_window_2D, unroll_window_3D
from utils.device import get_free_device
from utils.logger import create_logger
from utils.metrics import MetricsResult
from utils.utils import str2bool
import time

from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
hyp_kernel_size = 3
hyp_blocks = 2
hyp_layers = 2
hyp_pre_window = 4


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    # def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
    def __init__(self, file_name, config, graph_init):
        super(gwnet, self).__init__()
        self.dataset = config.dataset
        self.file_name = file_name

        device = torch.device(args.device)
        static_feat = None

        self.dropout = config.dropout
        self.gcn_bool = config.gcn_bool
        self.addaptadj = config.addaptadj
        self.graph_init = torch.from_numpy(graph_init)
        self.in_dim = config.in_dim
        self.out_dim = config.out_dim
        self.rolling_size = config.rolling_size
        self.num_nodes = config.num_nodes
        self.residual_channels = config.h_dim
        self.dilation_channels = config.h_dim
        self.skip_channels = 32
        self.end_channels = 32
        self.kernel_size = hyp_kernel_size
        self.blocks = hyp_blocks
        self.layers = hyp_layers
        self.dropout = config.dropout
        self.continue_training = config.continue_training
        self.robustness = config.robustness
        # pid
        self.pid = config.pid
        self.epochs = config.epochs
        self.display_epoch = config.display_epoch
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.milestone_epochs = config.milestone_epochs
        self.batch_size = config.batch_size
        self.load_model = config.load_model
        self.early_stopping = config.early_stopping
        self.gamma = config.gamma
        self.preprocessing = config.preprocessing
        self.use_overlapping = config.use_overlapping
        self.continue_training = config.continue_training

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_model/{}/'.format(self.dataset)):
                os.makedirs('./save_model/{}/'.format(self.dataset))
            self.save_model_path = \
                './save_model/{}/GWN_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem,
                                       self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_model/{}/GWN_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem,
                                       self.pid)
        else:
            self.load_model_path = None

        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        addaptadj = True
        aptinit = torch.from_numpy(graph_init)
        self.supports = []
        receptive_field = 1

        self.supports_len = 0

        if self.gcn_bool and addaptadj:
            if aptinit is None:
                exit()
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 8).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(8, self.num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :8], torch.diag(p[:8] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:8] ** 0.5), n[:, :8].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(self.dilation_channels, self.residual_channels, self.dropout,
                                          support_len=self.supports_len))

        file_logger.info("receptive_field: ", int(receptive_field))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels * self.num_nodes // 32,
                                    kernel_size=(self.num_nodes, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels * self.num_nodes // 32,
                                    out_channels=self.out_dim * self.num_nodes,
                                    kernel_size=(1, 1),
                                    bias=True)
        '''
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        '''
        self.receptive_field = receptive_field

    def forward(self, input):
        input = input.unsqueeze(3)
        input = input.transpose(1, 3)
        input = nn.functional.pad(input, (1, 0, 0, 0))
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        # file_logger.info("input size: ", x.size())
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.start_conv(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    exit()
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = nn.LeakyReLU(0.01)(skip)
        x = nn.LeakyReLU(0.01)(self.end_conv_1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.end_conv_2(x)
        # x = x.transpose(1,3) [:,:,:,0]
        # file_logger.info("output size: ", x.size())
        x = torch.mean(x, dim=-1, keepdim=False)
        x = torch.reshape(x, shape=[-1, self.out_dim, self.num_nodes])
        return x

    def fit(self, train_input, train_label, valid_input, valid_label, test_input, test_label, abnormal_data,
            abnormal_label, original_x_dim):
        loss_fn = nn.MSELoss()
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)
        # get batch data
        # if self.preprocessing == False or self.use_overlapping == False:
        #     train_input, test_input, test_label = np.expand_dims(np.transpose(train_input), axis=0), np.expand_dims(np.transpose(test_input), axis=0), np.expand_dims(test_label, axis=0)
        train_data = get_loader(input=train_input, label=train_input[:, self.rolling_size-self.out_dim:, :], batch_size=self.batch_size, from_numpy=True,
                                drop_last=False, shuffle=True)
        valid_data = get_loader(input=valid_input, label=valid_label[:, self.rolling_size-self.out_dim:, :], batch_size=self.batch_size, from_numpy=True,
                                drop_last=False, shuffle=False)
        test_data = get_loader(input=test_input, label=test_label, batch_size=self.batch_size, from_numpy=True,
                               drop_last=False, shuffle=False)
        min_valid_loss, all_patience, cur_patience, best_epoch = 1e20, 2, 1, 0
        if self.load_model == True and self.continue_training == False:
            exit()
            epoch_valid_losses = [-1]
            self.load_state_dict(torch.load(self.load_model_path))
        elif self.load_model == True and self.continue_training == True:
            exit()
            self.load_state_dict(torch.load(self.load_model_path))
            # train model
            epoch_losses = []
            epoch_valid_losses = []
            for epoch in range(self.epochs):
                self.train()
                train_losses = []
                # opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_x_reconstruct = self.forward(batch_x)
                    batch_loss = loss_fn(batch_x_reconstruct, batch_x)
                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))

                valid_losses = []
                # opt.zero_grad()
                self.eval()
                with torch.no_grad():
                    for i, (val_batch_x, val_batch_y) in enumerate(valid_data):
                        val_batch_x = val_batch_x.to(device)
                        val_batch_x_reconstruct = self.forward(val_batch_x)
                        val_batch_loss = loss_fn(val_batch_x_reconstruct, val_batch_x)
                        valid_losses.append(val_batch_loss.item())
                epoch_valid_losses.append(mean(valid_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , valid loss = {}'.format(epoch, epoch_valid_losses[-1]))

                if self.early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
        else:
            # train model

            epoch_losses = []
            epoch_valid_losses = []
            training_time = []
            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                self.train()
                train_losses = []
                # opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_x_reconstruct = self.forward(batch_x)
                    '''
                    predict_y = np.sum((batch_x_reconstruct - batch_x).detach().cpu().numpy() ** 2, axis=2,
                                       keepdims=False)
                    '''
                    batch_loss = loss_fn(batch_x_reconstruct, batch_y)
                    batch_loss.backward()
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))

                valid_losses = []
                # opt.zero_grad()
                self.eval()
                with torch.no_grad():
                    for i, (val_batch_x, val_batch_y) in enumerate(valid_data):
                        val_batch_x = val_batch_x.to(device)
                        val_batch_y = val_batch_y.to(device)
                        val_batch_x_reconstruct = self.forward(val_batch_x)
                        val_batch_loss = loss_fn(val_batch_x_reconstruct, val_batch_y)
                        valid_losses.append(val_batch_loss.item())
                epoch_valid_losses.append(mean(valid_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , valid loss = {}'.format(epoch, epoch_valid_losses[-1]))

                if self.early_stopping:
                    if len(epoch_valid_losses) > 1:
                        if epoch_valid_losses[best_epoch] - epoch_valid_losses[-1] < 1e-5:
                            train_logger.info('EarlyStopping counter: {} out of {}'.format(cur_patience, all_patience))
                            if cur_patience == all_patience:
                                train_logger.info('Early Stopping!')
                                break
                            cur_patience += 1
                        else:
                            train_logger.info("Saving Model.")
                            torch.save(self.state_dict(), self.save_model_path)
                            best_epoch = epoch
                            cur_patience = 1
                    else:
                        torch.save(self.state_dict(), self.save_model_path)
                epoch_end_time = time.time()
                training_time.append(epoch_end_time - epoch_start_time)

        training_time = np.mean(training_time)
        min_valid_loss = min(epoch_valid_losses)
        self.load_state_dict(torch.load(self.save_model_path))
        # test model
        self.eval()
        test_start_time = time.time()
        with torch.no_grad():
            cat_xs = []
            cat_ys = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                batch_y = batch_y[:, -1, :]
                batch_x = batch_x.to(device)
                batch_x_reconstruct = self.forward(batch_x)
                predict_y = np.sum((batch_x_reconstruct - batch_x[:, self.rolling_size-self.out_dim:, :]).detach().cpu().numpy() ** 2, axis=2, keepdims=False)
                predict_y = np.sum(predict_y, axis=1, keepdims=True)

                cat_xs.append(predict_y)
                cat_ys.append(batch_y.detach().cpu())

            test_end_time = time.time()
            cat_xs = np.concatenate(cat_xs, axis=0)
            cat_ys = np.concatenate(cat_ys, axis=0)
            testing_time = test_end_time - test_start_time
            gwn_output = GWNOutput(dec_means=[cat_xs, cat_ys], best_TN=None, best_FP=None, best_FN=None, best_TP=None,
                                   best_precision=None, best_recall=None, best_fbeta=None,
                                   best_pr_auc=None, best_roc_auc=None, best_cks=None, min_valid_loss=min_valid_loss,
                                   memory_usage_nvidia=None, training_time=training_time, testing_time=testing_time)
            return gwn_output


def RunModel(train_filename, test_filename, label_filename, config, ratio):
    # negative_sample = True if "noise" in config.dataset else False
    train_data, abnormal_data, abnormal_label, target_label = read_dataset(train_filename, test_filename,
                                                                           label_filename,
                                                                           normalize=True, file_logger=file_logger,
                                                                           negative_sample=False, ratio=ratio,
                                                                           pre_window=hyp_pre_window)
    abnormal_label = target_label
    standsacle = preprocessing.StandardScaler()
    standsacle.fit(train_data[:-1])
    graph_train_data = standsacle.transform(train_data[:-1], copy=True)
    cov_init = EmpiricalCovariance(store_precision=True, assume_centered=True).fit(graph_train_data)
    #cov_init = GraphicalLasso(alpha=.001,  mode='cd', tol=1e-4, enet_tol=1e-4, max_iter=100, assume_centered=True).fit(train_data)

    adj_mx_init = np.abs(cov_init.precision_)
    d_init = np.array(adj_mx_init.sum(1))
    d_inv = np.power(d_init, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)
    graph_matrix_init = d_mat_inv.dot(adj_mx_init).dot(d_mat_inv).astype(dtype='float32')
    file_logger.info("graph_matrix_init: ", graph_matrix_init)

    if abnormal_data.shape[0] < config.rolling_size:
        train_logger.warning("test data is less than rolling_size! Ignore the current data!")
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in ["SD", "MAD", "IQR"]:
            TN[threshold_method] = -1
            FP[threshold_method] = -1
            FN[threshold_method] = -1
            TP[threshold_method] = -1
            precision[threshold_method] = -1
            recall[threshold_method] = -1
            f1[threshold_method] = -1
        roc_auc = -1
        pr_auc = -1
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision,
                                       recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc)
        return metrics_result

    original_x_dim = abnormal_data.shape[1]

    rolling_train_data = None
    rolling_valid_data = None
    if config.preprocessing:
        if config.use_overlapping:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(train_data,
                                                                                                      config.rolling_size), rolling_window_2D(
                    abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
                train_split_idx = int(rolling_train_data.shape[0] * 0.7)
                rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[
                                                                                               train_split_idx:]
            else:
                exit()
                rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(abnormal_data,
                                                                                  config.rolling_size), rolling_window_2D(
                    abnormal_label, config.rolling_size)
        else:
            exit()
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(train_data,
                                                                                                      config.rolling_size), cutting_window_2D(
                    abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
                train_split_idx = int(rolling_train_data.shape[0] * 0.7)
                rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[
                                                                                               train_split_idx:]
            else:
                rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(abnormal_data,
                                                                                  config.rolling_size), cutting_window_2D(
                    abnormal_label, config.rolling_size)
    else:
        exit()
        if train_data is not None:
            rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = np.expand_dims(train_data,
                                                                                               axis=0), np.expand_dims(
                abnormal_data, axis=0), np.expand_dims(abnormal_label, axis=0)
            train_split_idx = int(rolling_train_data.shape[0] * 0.7)
            rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[
                                                                                           train_split_idx:]
        else:
            rolling_abnormal_data, rolling_abnormal_label = np.expand_dims(abnormal_data, axis=0), np.expand_dims(
                abnormal_label, axis=0)

    config.x_dim = rolling_abnormal_data.shape[1]

    model = gwnet(file_name=train_filename, config=config, graph_init=graph_matrix_init)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("total_params:", int(total_params))
    file_logger.info('============================')
    file_logger.info("total_params:", total_params)
    gwn_output = None
    if train_data is not None and config.robustness == False:
        gwn_output = model.fit(train_input=rolling_train_data, train_label=rolling_train_data,
                               valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                               original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        exit()
        gwn_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_data,
                               valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                               original_x_dim=original_x_dim)
    '''
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.01, 1.01))
    if config.preprocessing:
        if config.use_overlapping:
            if config.use_last_point:
                exit()
                dec_mean_unroll = gwn_output.dec_means.detach().cpu().numpy()[:, -1]
                #dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[config.rolling_size - 1:]
            else:
                dec_mean_unroll = unroll_window_3D(
                    np.reshape(gwn_output.dec_means.detach().cpu().numpy(), (-1, config.rolling_size, original_x_dim)))[
                                  ::-1]
                #dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]

        else:
            exit()
            dec_mean_unroll = np.reshape(gwn_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
            #dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
    else:
        exit()
        dec_mean_unroll = gwn_output.dec_means.detach().cpu().numpy()
        dec_mean_unroll = np.transpose(np.squeeze(dec_mean_unroll, axis=0))
        #dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
        x_original_unroll = abnormal_data
    '''

    error = gwn_output.dec_means[0]
    abnormal_label = gwn_output.dec_means[1]
    if config.save_output:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Dec_GWN_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.h_dim,
                                                                                       config.rolling_size,
                                                                                       Path(train_filename).stem,
                                                                                       config.pid),
                [error, abnormal_label])

    # error = np.sum((x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim])) ** 2, axis=1)
    # final_zscore = zscore(error)
    # np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
    # np_decision = create_label_based_on_quantile(error, quantile=99)
    SD_Tmin, SD_Tmax = SD_autothreshold(error)
    SD_y_hat = get_labels_by_threshold(error, Tmax=SD_Tmax, use_max=True, use_min=False)
    MAD_Tmin, MAD_Tmax = MAD_autothreshold(error)
    MAD_y_hat = get_labels_by_threshold(error, Tmax=MAD_Tmax, use_max=True, use_min=False)
    IQR_Tmin, IQR_Tmax = IQR_autothreshold(error)
    IQR_y_hat = get_labels_by_threshold(error, Tmax=IQR_Tmax, use_max=True, use_min=False)
    np_decision = {}
    np_decision["SD"] = SD_y_hat
    np_decision["MAD"] = MAD_y_hat
    np_decision["IQR"] = IQR_y_hat

    # TODO metrics computation.

    # %%
    if config.save_figure:
        file_logger.info('save_figure has been dropped.')

    if config.use_spot:
        pass
    else:
        pos_label = -1
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in np_decision:
            # cm = confusion_matrix(y_true=abnormal_label, y_pred=np_decision[threshold_method], labels=[1, -1])
            file_logger.info('============================')
            file_logger.info("abnormal_label len:", abnormal_label.shape[0])
            file_logger.info("np_decision len:", SD_y_hat.shape[0])
            cm = confusion_matrix(y_true=abnormal_label[:min(SD_y_hat.shape[0], abnormal_label.shape[0])],
                                  y_pred=np_decision[threshold_method], labels=[1, -1])
            TN[threshold_method] = cm[0][0]
            FP[threshold_method] = cm[0][1]
            FN[threshold_method] = cm[1][0]
            TP[threshold_method] = cm[1][1]
            #     precision[threshold_method] = precision_score(y_true=abnormal_label, y_pred=np_decision[threshold_method],
            #                                                   pos_label=pos_label)
            #     recall[threshold_method] = recall_score(y_true=abnormal_label, y_pred=np_decision[threshold_method],
            #                                             pos_label=pos_label)
            #     f1[threshold_method] = f1_score(y_true=abnormal_label, y_pred=np_decision[threshold_method],
            #                                     pos_label=pos_label)
            #
            # fpr, tpr, _ = roc_curve(y_true=abnormal_label, y_score=np.nan_to_num(error), pos_label=pos_label)
            # roc_auc = auc(fpr, tpr)
            # pre, re, _ = precision_recall_curve(y_true=abnormal_label, probas_pred=np.nan_to_num(error),
            #                                     pos_label=pos_label)
            precision[threshold_method] = precision_score(
                y_true=abnormal_label[:min(SD_y_hat.shape[0], abnormal_label.shape[0])],
                y_pred=np_decision[threshold_method], pos_label=pos_label)
            recall[threshold_method] = recall_score(
                y_true=abnormal_label[:min(SD_y_hat.shape[0], abnormal_label.shape[0])],
                y_pred=np_decision[threshold_method],
                pos_label=pos_label)
            f1[threshold_method] = f1_score(y_true=abnormal_label[:min(SD_y_hat.shape[0], abnormal_label.shape[0])],
                                            y_pred=np_decision[threshold_method],
                                            pos_label=pos_label)
        fpr, tpr, _ = roc_curve(y_true=abnormal_label[:min(SD_y_hat.shape[0], abnormal_label.shape[0])],
                                y_score=np.nan_to_num(error),
                                pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label[:min(SD_y_hat.shape[0], abnormal_label.shape[0])],
                                            probas_pred=np.nan_to_num(error), pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision,
                                       recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc,
                                       best_TN=gwn_output.best_TN, best_FP=gwn_output.best_FP,
                                       best_FN=gwn_output.best_FN, best_TP=gwn_output.best_TP,
                                       best_precision=gwn_output.best_precision, best_recall=gwn_output.best_recall,
                                       best_fbeta=gwn_output.best_fbeta, best_pr_auc=gwn_output.best_pr_auc,
                                       best_roc_auc=gwn_output.best_roc_auc, best_cks=gwn_output.best_cks,
                                       min_valid_loss=gwn_output.min_valid_loss, testing_time=gwn_output.testing_time,
                                       training_time=gwn_output.training_time, total_params=total_params)
        return metrics_result


if __name__ == '__main__':
    conn = sqlite3.connect('./experiments.db')
    cursor_obj = conn.cursor()

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--in_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=4)
    parser.add_argument('--out_dim', type=int, default=16)
    parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
    parser.add_argument('--gcn_bool', action='store_true', default=True, help='whether to add graph convolution layer')
    parser.add_argument('--addaptadj', action='store_true', default=True, help='whether add adaptive adj')
    parser.add_argument('--num_nodes', type=int, default=51, help='number of nodes/variables')
    parser.add_argument('--device', type=str, default='cuda', help='device to run')
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--rolling_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--early_stopping', type=str2bool, default=True)
    parser.add_argument('--loss_function', type=str, default='mse')
    parser.add_argument('--display_epoch', type=int, default=1)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=False)
    parser.add_argument('--save_model', type=str2bool, default=True)  # save model
    parser.add_argument('--save_results', type=str2bool, default=True)  # save results
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--continue_training', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--use_last_point', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=False)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--pid', type=str, default='0')
    args = parser.parse_args()

    # for registered_dataset in ["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    # for registered_dataset in ["Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    # for registered_dataset in ["SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    # for registered_dataset in ["SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    # for registered_dataset in ["MSL"]:
    for registered_dataset in ["SWAT"]:

        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset
        args.pid = args.pid + '_' + time.strftime('%m%d%H%M%S')
        if args.load_config:
            config = GWNConfig(dataset=None, in_dim=None, h_dim=None, out_dim=None, adjdata=None, adjtype=None,
                               gcn_bool=None,
                               num_nodes=None, device=None, preprocessing=None, addaptadj=None, use_overlapping=None,
                               rolling_size=None, epochs=None, milestone_epochs=None, lr=None, gamma=None,
                               batch_size=None,
                               weight_decay=None, early_stopping=None, loss_function=None, display_epoch=None,
                               save_output=None, save_figure=None, save_model=None, load_model=None,
                               continue_training=None,
                               dropout=None, use_spot=None, use_last_point=None, save_config=None, load_config=None,
                               server_run=None, robustness=None, pid=None, save_results=None)
            try:
                config.import_config('./config/{}/Config_GWN_pid={}.json'.format(config.dataset, config.pid))
            except:
                print('There is no config.')
        else:
            config = GWNConfig(dataset=args.dataset, in_dim=args.in_dim, h_dim=args.h_dim, out_dim=args.out_dim,
                               adjdata=args.adjdata,
                               adjtype=args.adjtype, gcn_bool=args.gcn_bool, num_nodes=args.num_nodes,
                               addaptadj=args.addaptadj,
                               device=args.device, preprocessing=args.preprocessing,
                               use_overlapping=args.use_overlapping, rolling_size=args.rolling_size, epochs=args.epochs,
                               milestone_epochs=args.milestone_epochs, lr=args.lr, gamma=args.gamma,
                               batch_size=args.batch_size, weight_decay=args.weight_decay,
                               early_stopping=args.early_stopping, loss_function=args.loss_function,
                               display_epoch=args.display_epoch, save_output=args.save_output,
                               save_figure=args.save_figure,
                               save_model=args.save_model, load_model=args.load_model,
                               continue_training=args.continue_training, dropout=args.dropout, use_spot=args.use_spot,
                               use_last_point=args.use_last_point, save_config=args.save_config,
                               load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                               pid=args.pid, save_results=args.save_results)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_GWN_pid={}.json'.format(config.dataset, args.pid))
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]
        # %%
        # device = torch.device(get_free_device())
        args.device = torch.device("cuda")
        device = torch.device(args.device)

        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               h_dim=config.h_dim,
                                                               rolling_size=config.rolling_size,
                                                               train_logger_name='gwn_train_logger',
                                                               file_logger_name='gwn_file_logger',
                                                               meta_logger_name='gwn_meta_logger',
                                                               model_name='GWN',
                                                               pid=args.pid)

        # logging setting
        file_logger.info('============================')
        for key, value in vars(args).items():
            file_logger.info(key + ' = {}'.format(value))
        file_logger.info('============================')

        meta_logger.info('============================')
        for key, value in vars(args).items():
            meta_logger.info(key + ' = {}'.format(value))
        meta_logger.info('============================')

        # for train_file in train_path.iterdir():
        for train_file in train_path.iterdir():
            # if train_file.name != "real_62.pkl":
            #    continue
            # for train_file in [Path('../datasets/train/MSL/M-1.pkl')]:
            test_file = test_path / train_file.name
            label_file = label_path / train_file.name
            file_logger.info('============================')
            file_logger.info(train_file)

            metrics_result = RunModel(train_filename=train_file, test_filename=test_file, label_filename=label_file,
                                      config=config, ratio=args.ratio)
            result_dataframe = make_result_dataframe(metrics_result)

            if config.save_results == True:
                if not os.path.exists('./results/{}/'.format(config.dataset)):
                    os.makedirs('./results/{}/'.format(config.dataset))
                result_dataframe.to_csv(
                    './results/{}/EACH_hdim{}_roll{}_out{}_{}_b{}-l{}-k{}-pre{}_pid={}.csv'.format(config.dataset,
                                                                                                     config.h_dim,
                                                                                                     config.rolling_size,
                                                                                                   config.out_dim,
                                                                                                     train_file.stem,
                                                                                                     hyp_blocks,
                                                                                                     hyp_layers,
                                                                                                     hyp_kernel_size,
                                                                                               hyp_pre_window,
                                                                                                     config.pid),
                    index=False)
