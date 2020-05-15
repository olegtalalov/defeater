
import sys
import os
import shutil
import random
import zipfile
import tarfile
import numpy as np
import pandas as pd

def fill_with_empty_lables(waymo_path, label_folder, image_folder):
    """Fills label_folder files with empty label .txt files for frustum-net name consistency with image_folder .png files"""
    existing_label_names = set(map(lambda fn: os.path.splitext(fn)[0],
                                    os.listdir(os.path.join(waymo_path, label_folder))))

    required_names = set(map(lambda fn: os.path.splitext(fn)[0],
                             os.listdir(os.path.join(waymo_path, image_folder))))

    for empty_labels in (required_names - existing_label_names):
        open(os.path.join(waymo_path, label_folder, empty_labels + ".txt"), "a").close()

def extract_from_tar(source_path, dest_path):
    file_tar = tarfile.open(name=source_path, mode='r', fileobj=None, bufsize=10240)
    file_tar.extractall(path=dest_path)

def zip_dir(source_path_list, dest_path):
    temp_dir = "/content/temp_dir"
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    for source_path in source_path_list:
        zipf = zipfile.ZipFile(os.path.join(temp_dir, 'zipped_{0}.zip').format(os.path.basename(source_path)),
                                'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        for root, dirs, files in os.walk(source_path):
            for file_to_zip in files:
                zipf.write(os.path.join(root, file_to_zip))
        zipf.close()
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)
        for zip_file in os.listdir(temp_dir):
            shutil.copy(os.path.join(temp_dir, zip_file), dest_path)


def split_to_tracks(metadata_path, source_path, dest_path):
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    grouped_metadata = pd.DataFrame(pd.read_csv(metadata_path).groupby("track_id").agg("point_cloud_id"),
                                    columns=["track_id", "bbox_ids"])

    for _, row in grouped_metadata.iterrows():
        if not os.path.exists(os.path.join(dest_path, row['track_id'])):
            os.mkdir(os.path.join(dest_path, row['track_id']))

        for i in row['bbox_ids']:
            shutil.copy(
                        os.path.join(source_path, (str(i) + ".npy").zfill(12) ),
                        os.path.join(dest_path, row['track_id'])
                       )

def merge_fp(fp_dir, dest_path):
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    fp_detections = [os.path.join(dp, f) for dp, _, fn in os.walk(os.path.expanduser(fp_dir)) for f in fn]

    for i, fp_det in enumerate(fp_detections):
        shutil.copyfile(fp_det, os.path.join(dest_path, str(i).zfill(12) + ".npy"))
