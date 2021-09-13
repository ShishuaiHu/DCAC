# -*- coding:utf-8 _*-
# @author: sshu
# @contact: sshu@mail.nwpu.edu.cn
# @version: 0.1.0
# @file: Task1001_Prostate.py
# @time: 2021/09/13
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk


def process_prostate_seg(img_path, seg_path, saved_img_path, saved_seg_path):
    img_sitk = sitk.ReadImage(img_path)
    seg_sitk = sitk.ReadImage(seg_path)

    img_npy = sitk.GetArrayFromImage(img_sitk)
    seg_npy = sitk.GetArrayFromImage(seg_sitk)
    seg_npy[seg_npy > 0] = 1.0

    for i in range(seg_npy.shape[0]):
        if seg_npy[i, :, :].max() > 0:
            start_slice = i
            break
    for i in range(seg_npy.shape[0])[::-1]:
        if seg_npy[i, :, :].max() > 0:
            end_slice = i+1
            break
    new_seg_npy = seg_npy[start_slice:end_slice, :, :]
    new_img_npy = img_npy[start_slice:end_slice, :, :]
    new_seg_sitk = sitk.GetImageFromArray(new_seg_npy)
    new_img_sitk = sitk.GetImageFromArray(new_img_npy)
    new_seg_sitk.SetDirection(seg_sitk.GetDirection())
    new_seg_sitk.SetSpacing(seg_sitk.GetSpacing())
    new_seg_sitk.SetOrigin(seg_sitk.GetOrigin())

    new_img_sitk.SetDirection(img_sitk.GetDirection())
    new_img_sitk.SetSpacing(img_sitk.GetSpacing())
    new_img_sitk.SetOrigin(img_sitk.GetOrigin())

    sitk.WriteImage(new_img_sitk, saved_img_path)
    sitk.WriteImage(new_seg_sitk, saved_seg_path)


if __name__ == "__main__":
    base = "Multi-site Prostate MRI Segmentation"  # The path where the downloaded prostate segmentation dataset placed

    task_id = 1001
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
        test_patients = subfiles(join(base, target_domain_folder), join=False, suffix='segmentation.nii.gz')
        test_case_id = 0
        for case in test_patients:
            p = target_domain_folder+'_{0:04}'.format(test_case_id)
            label_file = join(base, target_domain_folder, case)
            image_file = join(base, target_domain_folder, case.replace('_segmentation', ''))
            process_prostate_seg(image_file, label_file, join(imagests, p + "_0000.nii.gz"),
                                 join(labelsts, p + ".nii.gz"))
            test_patient_names.append(p)
            test_case_id += 1

        # copy to source_train_folder
        train_patients = list()
        train_case_id = 0
        for source_domain_folder in source_domain_folders:
            train_patients = subfiles(join(base, source_domain_folder), join=False, suffix='segmentation.nii.gz')
            for case in train_patients:
                p = source_domain_folder + '_{0:04}'.format(train_case_id)
                label_file = join(base, source_domain_folder, case)
                image_file = join(base, source_domain_folder, case.replace('_segmentation', ''))
                process_prostate_seg(image_file, label_file, join(imagestr, p + "_0000.nii.gz"),
                                     join(labelstr, p + ".nii.gz"))
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
            "0": "MR",
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "prostate",
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

