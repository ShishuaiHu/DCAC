# -*- coding:utf-8 _*-
# @author: sshu
# @contact: sshu@mail.nwpu.edu.cn
# @version: 0.1.0
# @file: DCAC_Prostate.py
# @time: 2021/09/13
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
import torch.nn.functional as F
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, StackedConvLayers, \
    Upsample, Generic_UNet


source_number = 5
segmentation_class = 2
hyper_k1 = source_number*segmentation_class
hyper_k2 = source_number*segmentation_class


class DCAC_Prostate(Generic_UNet):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
            self.input_type = '2D'
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
            self.input_type = '3D'
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []
        self.gap_blocks = []

        gap_features_num = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))

            if self.input_type == '3D':
                self.gap_blocks.append(nn.Sequential(
                    torch.nn.AdaptiveAvgPool3d((1, 1, 1))
                ))
            else:
                self.gap_blocks.append(nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d((1, 1))
                ))
            gap_features_num.append(output_features)

            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # Classifier & Global info
        if self.input_type == '3D':
            self.GAP = nn.Sequential(
                torch.nn.AdaptiveAvgPool3d((1, 1, 1))
            )
        else:
            self.GAP = nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )
        self.classifier_fc = nn.Sequential(
            nn.Linear(final_num_features + np.sum(gap_features_num), source_number)
        )
        da_patameters = hyper_k1 * hyper_k1 + hyper_k1
        if self.input_type == '3D':
            self.da_controller = nn.Conv3d(source_number, da_patameters, kernel_size=1, stride=1, padding=0)
        else:
            self.da_controller = nn.Conv2d(source_number, da_patameters, kernel_size=1, stride=1, padding=0)
        parameter_numbers = hyper_k1*hyper_k2 + hyper_k2*hyper_k2 + hyper_k2*num_classes + hyper_k2 + hyper_k2 + num_classes
        if self.input_type == '3D':
            self.controller = nn.Conv3d(final_num_features, parameter_numbers, kernel_size=1, stride=1, padding=0)
        else:
            self.controller = nn.Conv2d(final_num_features, parameter_numbers, kernel_size=1, stride=1, padding=0)

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            # self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
            #                                 1, 1, 0, 1, 1, seg_output_use_bias))
            self.seg_outputs.append(nn.Sequential(conv_op(self.conv_blocks_localization[ds][-1].output_channels, hyper_k1,
                                            1, 1, 0, 1, 1, bias=True),
                                    self.norm_op(hyper_k1, **self.norm_op_kwargs),
                                    self.nonlin(**self.nonlin_kwargs)))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

        self.classifier_out = None
        self.batch_dom_class = None

    def parse_dynamic_params(self, params, in_channels, out_channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                if self.input_type == '3D':
                    weight_splits[l] = weight_splits[l].reshape(num_insts * in_channels, -1, 1, 1, 1)
                else:
                    weight_splits[l] = weight_splits[l].reshape(num_insts * in_channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * in_channels)
            else:
                # out_channels x in_channels x 1 x 1
                if self.input_type == '3D':
                    weight_splits[l] = weight_splits[l].reshape(num_insts * out_channels, -1, 1, 1, 1)
                else:
                    weight_splits[l] = weight_splits[l].reshape(num_insts * out_channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * out_channels)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        if self.input_type == '3D':
            assert features.dim() == 5
        else:
            assert features.dim() == 4

        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            if self.input_type == '3D':
                x = F.conv3d(
                    x, w, bias=b,
                    stride=1, padding=0,
                    groups=num_insts
                )
            else:
                x = F.conv2d(
                    x, w, bias=b,
                    stride=1, padding=0,
                    groups=num_insts
                )
            if i < n_layers - 1:
                x = F.group_norm(x, num_groups=num_insts)
                x = F.leaky_relu(x)
        return x

    def forward(self, x):
        skips = []
        seg_outputs = []
        gap_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

            gap_in = x.detach()
            gap_out = self.gap_blocks[d](gap_in)
            gap_outputs.append(gap_out)

            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        global_f = self.GAP(x)
        global_f_c = global_f.detach()

        for gap_output in gap_outputs[::-1]:
            global_f_c = torch.cat((global_f_c, gap_output), dim=1)

        classifier_f = torch.flatten(global_f_c, 1)
        classifier_out = self.classifier_fc(classifier_f)
        self.classifier_out = classifier_out
        if self.batch_dom_class is None:
            batch_class = softmax_helper(classifier_out)
        else:
            batch_class = self.batch_dom_class
        # batch_class = batch_class.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        if self.input_type == '3D':
            batch_class = batch_class.view(batch_class.shape[0], batch_class.shape[1], 1, 1, 1)
        else:
            batch_class = batch_class.view(batch_class.shape[0], batch_class.shape[1], 1, 1)
        da_params = self.da_controller(batch_class)
        da_params = da_params.view(da_params.shape[0], da_params.shape[1])
        da_weight_nums, da_bias_nums = [], []
        da_weight_nums.append(hyper_k1*hyper_k1)
        da_bias_nums.append(hyper_k1)
        da_weights, da_biases = self.parse_dynamic_params(da_params, hyper_k1, hyper_k1, da_weight_nums, da_bias_nums)


        params = self.controller(global_f)
        # params = params.squeeze(-1).squeeze(-1).squeeze(-1)
        params = params.view(params.shape[0], params.shape[1])
        weight_nums, bias_nums = [], []
        weight_nums.append(hyper_k1 * hyper_k2)
        weight_nums.append(hyper_k2 * hyper_k2)
        weight_nums.append(hyper_k2 * self.num_classes)
        bias_nums.append(hyper_k2)
        bias_nums.append(hyper_k2)
        bias_nums.append(self.num_classes)
        weights, biases = self.parse_dynamic_params(params, hyper_k2, self.num_classes, weight_nums, bias_nums)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            # seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            head_inputs = self.seg_outputs[u](x)
            if self.input_type == '3D':
                N, _, D, H, W = head_inputs.size()
                head_inputs = head_inputs.reshape(1, -1, D, H, W)
                res_da_out = head_inputs
                da_out = self.heads_forward(head_inputs, da_weights, da_biases, N)
                da_out = F.group_norm(da_out, num_groups=N)
                da_out += res_da_out
                da_out = F.leaky_relu(da_out)
                logits = self.heads_forward(da_out, weights, biases, N)
                logits = logits.reshape(N, -1, D, H, W)
            else:
                N, _, D, W = head_inputs.size()
                head_inputs = head_inputs.reshape(1, -1, D, W)
                res_da_out = head_inputs
                da_out = self.heads_forward(head_inputs, da_weights, da_biases, N)
                da_out = F.group_norm(da_out, num_groups=N)
                da_out = res_da_out - da_out
                da_out = F.leaky_relu(da_out)
                logits = self.heads_forward(da_out, weights, biases, N)
                logits = logits.reshape(N, -1, D, W)
            seg_outputs.append(self.final_nonlin(logits))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
