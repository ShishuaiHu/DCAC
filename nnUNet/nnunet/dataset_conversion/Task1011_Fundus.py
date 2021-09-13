# -*- coding:utf-8 _*-
# @author: sshu
# @contact: sshu@mail.nwpu.edu.cn
# @version: 0.1.0
# @file: Task1011_Fundus.py
# @time: 2021/09/13
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
from skimage import io
import numpy as np


def convert_2d_image_to_nifti(input_image, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    img = input_image

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")


def process_image(image_file, target_dir, p):
    output_filename_truncated = join(target_dir, p)
    input_image = io.imread(image_file)
    convert_2d_image_to_nifti(input_image, output_filename_truncated, spacing=(1, 1, 1), transform=None, is_seg=False)


def process_label(label_file, target_dir, p):
    output_filename_truncated = join(target_dir, p)
    label_image = io.imread(label_file)
    input_image = np.zeros((label_image.shape[0], label_image.shape[1], 1))
    input_image[:, :, 0] = label_image[:, :, 0]
    input_image[input_image == 0] = 2
    input_image[input_image == 255] = 0
    input_image[input_image == 128] = 1
    assert (input_image >= 0).all() and (input_image <= 2).all()
    convert_2d_image_to_nifti(input_image, output_filename_truncated, spacing=(1, 1, 1), transform=None, is_seg=True)


if __name__ == "__main__":
    base = "Fundus"

    task_id = 1011
    target_domain_folders = subfolders(base, join=False)
    for target_domain_folder in target_domain_folders:
        source_domain_folders = target_domain_folders.copy()
        source_domain_folders.remove(target_domain_folder)
        task_name = 'Target_{}'.format(target_domain_folder)
        foldername = "Task%03.0d_%s" % (task_id, task_name)
        out_base = join(nnUNet_raw_data, foldername)
        imagestr = join(out_base, "imagesTr")
        imagests = join(out_base, "imagesTs")
        labelstr = join(out_base, "labelsTr")
        labelsts = join(out_base, "labelsTs")
        maybe_mkdir_p(imagestr)
        maybe_mkdir_p(imagests)
        maybe_mkdir_p(labelstr)
        maybe_mkdir_p(labelsts)

        train_patient_names = []
        test_patient_names = []

        # copy to target_test_tolder
        test_patients = subfiles(join(base, target_domain_folder, 'test', 'ROIs', 'image'), join=False)
        test_case_id = 0
        for case in test_patients:
            p = target_domain_folder+'_{0:04}'.format(test_case_id)
            label_file = join(base, target_domain_folder, 'test', 'ROIs', 'mask', case)
            image_file = join(base, target_domain_folder, 'test', 'ROIs', 'image', case)
            process_image(image_file, imagests, p)
            process_label(label_file, labelsts, p)
            test_patient_names.append(p)
            test_case_id += 1

        # copy to source_train_folder
        train_patients = list()
        train_case_id = 0
        for source_domain_folder in source_domain_folders:
            train_patients = subfiles(join(base, source_domain_folder, 'train', 'ROIs', 'image'), join=False)
            for case in train_patients:
                p = source_domain_folder + '_{0:04}'.format(train_case_id)
                label_file = join(base, source_domain_folder, 'train', 'ROIs', 'mask', case)
                image_file = join(base, source_domain_folder, 'train', 'ROIs', 'image', case)
                process_image(image_file, imagestr, p)
                process_label(label_file, labelstr, p)
                train_patient_names.append(p)
                train_case_id += 1

        json_dict = {}
        json_dict['name'] = task_name
        json_dict['description'] = task_name
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "Channel0",
            "1": "Channel1",
            "2": "Channel2"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "disk",
            "2": "cup"
        }

        json_dict['numTraining'] = len(train_patient_names)
        json_dict['numTest'] = len(test_patient_names)
        json_dict['training'] = [
            {'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for
            i in
            train_patient_names]
        json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

        save_json(json_dict, os.path.join(out_base, "dataset.json"))
        task_id += 1

