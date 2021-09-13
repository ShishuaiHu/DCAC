# -*- coding:utf-8 _*-
# @author: sshu
# @contact: sshu@mail.nwpu.edu.cn
# @version: 0.1.0
# @file: DCACTrainer_COVID.py
# @time: 2021/09/13
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import torch
from nnunet.network_architecture.DCAC.DCAC_COVID import DCAC_COVID
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from nnunet.DCAC.DCAC_encoding import *


class DCACTrainer_COVID(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.class_loss = nn.CrossEntropyLoss()
        self.max_num_epochs = 1000

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = DCAC_COVID(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        keys = data_dict['keys']
        batch_class, cond_encoding = DCAC_encoding_class_for_COVID(keys, self.output_folder)

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        cond_encoding = torch.from_numpy(cond_encoding).long()
        batch_class = torch.from_numpy(batch_class)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            cond_encoding = to_cuda(cond_encoding)
            batch_class = to_cuda(batch_class)

        self.optimizer.zero_grad()
        self.network.batch_dom_class = batch_class

        train_cls = True

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                if not train_cls:
                    l = self.loss(output, target)
                else:
                    classifier_out = self.network.classifier_out
                    l = self.class_loss(classifier_out, cond_encoding) + self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            if not train_cls:
                l = self.loss(output, target)
            else:
                classifier_out = self.network.classifier_out
                l = self.class_loss(classifier_out, cond_encoding) + self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        self.network.batch_dom_class = None

        return l.detach().cpu().numpy()
