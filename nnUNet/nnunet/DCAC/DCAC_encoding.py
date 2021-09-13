# -*- coding:utf-8 _*-
# @author: sshu
# @contact: sshu@mail.nwpu.edu.cn
# @version: 0.1.0
# @file: DCAC_encoding.py
# @time: 2021/09/13
import numpy as np


def DCAC_encoding_class_for_Prostate(keys, output_folder):
    cond_keys = ['HK', 'BIDMC', 'I2CVB', 'ISBI_1.5', 'ISBI', 'UCL']
    target_cond = '_'.join(output_folder.split('/')[-3].split('_')[2:])
    cond_keys.remove(target_cond)
    cond_id = np.zeros(shape=(len(keys))).astype(int)
    batch_class = np.zeros(shape=[len(keys), len(cond_keys)]).astype(np.float32)
    for kk in range(len(keys)):
        cond_name = '_'.join(keys[kk].split('_')[:-1])
        try:
            cond_id[kk] = cond_keys.index(cond_name)
            batch_class[kk][cond_id[kk]] = 1
        except:
            raise Exception('unknown case name! {}'.format(cond_name))
    return batch_class, cond_id


def DCAC_encoding_class_for_COVID(keys, output_folder):
    cond_keys = ['SITE-A', 'SITE-B', 'SITE-C', 'SITE-D']
    target_cond = ''.join(output_folder.split('/')[-3].split('_')[2:])
    cond_keys.remove(target_cond)
    cond_id = np.zeros(shape=(len(keys))).astype(int)
    batch_class = np.zeros(shape=[len(keys), len(cond_keys)]).astype(np.float32)
    for kk in range(len(keys)):
        cond_name = ''.join(keys[kk].split('_')[:-1])
        try:
            cond_id[kk] = cond_keys.index(cond_name)
            batch_class[kk][cond_id[kk]] = 1
        except:
            raise Exception('unknown case name! {}'.format(cond_name))
    return batch_class, cond_id


def DCAC_encoding_class_for_Fundus(keys, output_folder):
    cond_keys = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
    target_cond = '_'.join(output_folder.split('/')[-3].split('_')[2:])
    cond_keys.remove(target_cond)
    cond_id = np.zeros(shape=(len(keys))).astype(int)
    batch_class = np.zeros(shape=[len(keys), len(cond_keys)]).astype(np.float32)
    for kk in range(len(keys)):
        cond_name = '_'.join(keys[kk].split('_')[:-1])
        try:
            cond_id[kk] = cond_keys.index(cond_name)
            batch_class[kk][cond_id[kk]] = 1
        except:
            raise Exception('unknown case name! {}'.format(cond_name))
    return batch_class, cond_id
