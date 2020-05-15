import os
import argparse
import tqdm

import collections
import numpy as np
import pandas as pd
import tensorflow as tf

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset.utils import  box_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

class PointCloudParser(object):
    """Performs point cloud 3D-box filtering for a given set of classes."""

    def __init__(self, data_path, max_segments=None, cls_to_filter=("TYPE_PEDESTRIAN", "TYPE_VEHICLE")):
        """
            data_path: Data path to folder that containts Waymo TFRecords

            max_segments: number of segments to process (None eq. all segments)

            cls_to_filter: point clouds for these classes will be extracted
        """

        assert os.path.exists(data_path), "Existing path is required"

        assert np.all([os.path.splitext(file_name)[-1] == ".tfrecord"
                       for file_name in os.listdir(data_path)]), "All files in the folder should be .tfrecord"

        assert set(cls_to_filter) in {"TYPE_UNKNOWN", "TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_SIGN", "TYPE_CYCLIST"}, "Invalid object type to filter"

        self.data_path = data_path

        self.cls_to_filter = cls_to_filter

        self.segments_to_proc = os.listdir(self.data_path)[:max_segments]

        self.type_enum = {
                            "TYPE_UNKNOWN" : 0,
                            "TYPE_VEHICLE" : 1,
                            "TYPE_PEDESTRIAN" : 2,
                            "TYPE_SIGN" : 3,
                            "TYPE_CYCLIST" : 4
                         }

        self.code_2_type = {v:k for k,v in self.type_enum.items()}

        self.obj_codes = tuple(self.type_enum[ctf] for ctf in self.cls_to_filter)

        self.bbox_count_dict = collections.defaultdict(int)


    def start_processing(self, dest_path, metadata_path, path_to_drive=None, min_pts_threshold=75):
        """
        Main routine method that starts points filtering.
        Creates len(self.obj_codes) number folders in dest_path
        for each class and saves point clouds for each box in .npy binary file.
        """

        self.check_dir(dest_path, metadata_path)
        self.dest_path = dest_path
        self.metadata_path = metadata_path
        self.metadata_dict = {obj:[] for obj in self.obj_codes}
        self.bbox_dict = {obj:[] for obj in self.obj_codes}
        self.min_pts_threshold = min_pts_threshold

        for segment_num, segment_name in enumerate(self.segments_to_proc):

            print("Segment #{0}/{1}".format(segment_num, len(self.segments_to_proc)))
            self.segment_name = segment_name

            dataset = tf.data.TFRecordDataset(os.path.join(self.data_path, segment_name),
                                              compression_type="")

            self.filter_frame_points(dataset)

        self.save_point_cloud()
        self.save_box_metadata()

    def filter_frame_points(self, dataset):
        """
        Return dict that maps from object type ids to point clouds.
        Value for each key is a list of numpy arrays,
        where each numpy array containts points for a single 3D bbox.
        """
        for frame_num, data in tqdm.tqdm(enumerate(dataset)):

            self.frame_num = frame_num

            frame = open_dataset.Frame()

            frame.ParseFromString(bytearray(data.numpy()))

            frame_pts = self.frame_points(frame)

            for obj_code in self.obj_codes:
                boxes, metadata = self.frame_boxes_3D(frame, obj_code)

            if len(boxes):
                for i, mask in enumerate(tf.transpose(box_utils.is_within_box_3d(frame_pts, np.array(boxes)))):
                    points_number = mask.numpy().sum()
                    if points_number >= self.min_pts_threshold:
                        self.bbox_dict[obj_code].append(tf.boolean_mask(frame_pts, mask).numpy())
                        metadata[i].append(points_number)
                        self.metadata_dict[obj_code].append(np.array(metadata[i]))


    def frame_points(self, frame, projection=False):
        """
        Returns Tensor of points in a vehicle coord. system for a given frame.
        If projection is True => returns points projection on frame
        """
        (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                                                                            frame,
                                                                            range_images,
                                                                            camera_projections,
                                                                            range_image_top_pose,
                                                                            ri_index=0
                                                                           )

        if projection:
            return tf.constant(np.concatenate(cp_points))
        else:
            return tf.constant(np.concatenate(points, axis=0))


    def frame_boxes_3D(self, frame, obj_code):
        """
        Returns np.array of box parameters for a given frame
        """

        box_stats = []
        metadata_stats = []

        for label in frame.laser_labels:
            if label.type == obj_code:
                box_stats.append(
                                 tf.constant([label.box.center_x, label.box.center_y, label.box.center_z,
                                              label.box.width, label.box.length, label.box.height,
                                              label.box.heading])
                                )

                metadata_stats.append(
                                      [
                                      self.segment_name,                  # segment_name
                                      self.frame_num,                     # frame number
                                      label.id,                           # track id
                                      np.linalg.norm([label.box.center_x, # polar coord. of box in vehicle frame (distance and angle)
                                                      label.box.center_y,
                                                      label.box.center_z]),
                                      np.arctan2(label.box.center_y,
                                                  label.box.center_x)
                                      ]
                                     )

        return box_stats, metadata_stats


    def check_dir(self, dest_path, metadata_path):
        """Creates required dir. for files"""
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)

        for obj_code in self.obj_codes:
            obj_folder = os.path.join(dest_path, self.code_2_type[obj_code])
            if not os.path.exists(obj_folder):
                os.mkdir(obj_folder)

        if not os.path.exists(metadata_path):
            os.mkdir(metadata_path)


    def save_point_cloud(self):
        """Saves filtered point cloud (numpy array of size [N, 3]) into .npy binary file."""

        print("Saving...")

        for obj_code in self.bbox_dict.keys():
            for i, bbox in tqdm.tqdm(enumerate(self.bbox_dict[obj_code])):
                np.save(
                          os.path.join(self.dest_path, self.code_2_type[obj_code],
                                      "{0}.npy".format(i).zfill(12)),
                          bbox
                       )

    def save_box_metadata(self):
        """
        Save box metadata from metadata_dict. Creates metadata csv file for every obj_type
        Box metadata includes box track_id and polar coord. to box in vehicle coord. system
        """
        print("Saving metadata...")
        for obj_code in self.metadata_dict.keys():
            metadata_df = pd.DataFrame(self.metadata_dict[obj_code])
            metadata_df.columns = ["segment_name", "frame_num", "track_id", "box_dist", "box_angle", "number_of_points"]
            metadata_df["point_cloud_id"] = np.arange(0, len(self.metadata_dict[obj_code]))
            if metadata_df.to_csv(os.path.join(self.metadata_path, r"metadata_{0}.csv".format(self.code_2_type[obj_code])),
                             index = False, header=True):
                print("Metadata saved successfully!")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_path', required=True, help='Source path')
    parser.add_argument('--dest_path', required=True, help='Destination path for bounding boxes')
    parser.add_argument('--metadata_dest_path', required=True, help='Destination path to metadata table')
    parser.add_argument('--max_segments', type=int, required=True, help='Max segments to process')
    parser.add_argument('--classes', nargs='*', help='Object classes to parse')
    parser.add_argument('--min_points_threshold', type=int, default=1, help='Minimun points in bounding box')

    args = parser.parse_args()

    point_cloud_parser = PointCloudParser(
                                          data_path=args.source_path,
                                          max_segments=arg.max_segments,
                                          cls_to_filter=args.classes
    )

    point_cloud_parser.start_processing(
                                        dest_path=args.dest_path,
                                        metadata_path=args.metadata_dest_path,
                                        min_pts_threshold=args.min_points_threshold
    )


if __name__ == "__main__":
    main()
