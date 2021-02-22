# Adapted version of https://github.com/Yao-Shao/Waymo_Kitti_Adapter conversion program
# This version produces not purely KITTI-style data format, and has some useful
# additional fields (at least for me, e.g. no label zone tags in point cloud data)

from __future__ import absolute_import, print_function, division
import sys
import os
import math
import argparse
import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

############################Config###########################################
# path to waymo dataset "folder" (all .tfrecord files in that folder will be converted)
# Metavars
LOCATION_FILTER_FLAG = False
LOCATION_NAME = {'location_sf'}
INDEX_LENGTH = 12
IMAGE_FORMAT = 'png'

###############################################################################

class Adapter(object):
    def __init__(self, source_dir, target_dir):

        self.__lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        self.__type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

        self.camera_id = 0

        self.DATA_PATH = source_dir
        self.KITTI_PATH = target_dir

        self.LABEL_PATH = self.KITTI_PATH + '/label'
        self.LABEL_ALL_PATH = self.KITTI_PATH + '/label_all'
        self.IMAGE_PATH = self.KITTI_PATH + '/image'
        self.CALIB_PATH = self.KITTI_PATH + '/calib'
        self.LIDAR_PATH = self.KITTI_PATH + '/velodyne'

        self.get_file_names()
        self.create_folder()

        print("Initializing is complete.")

    def cvt(self):
        """
        convert dataset from Waymo to KITTI
        Args:
        return:
        """
        tf.enable_eager_execution()
        file_num = 1
        frame_num = 1

        print("Start converting ...")
        for file_name in tqdm.tqdm(self.__file_names):
            dataset = tf.data.TFRecordDataset(file_name, compression_type='')
            for data in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                if LOCATION_FILTER_FLAG and frame.context.stats.location not in LOCATION_NAME:
                    continue

                self.save_image(frame, frame_num)
                self.save_calib(frame, frame_num)
                self.save_lidar(frame, frame_num)
                self.save_label(frame, frame_num)

                print("Frame number {0}".format(frame_num))
                frame_num += 1

            print("File number {0}".format(file_num))
            file_num += 1

        print("\nFinished!")

    def save_image(self, frame, frame_num):
        """
         parse and save the images in png format
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
        """
        print("Image saving routine")
        for img in frame.images:
            img_path = self.IMAGE_PATH + "_" + str(img.name - 1) + '/' + str(frame_num).zfill(INDEX_LENGTH) + '.' + IMAGE_FORMAT
            img = cv2.imdecode(np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.imsave(img_path, rgb_img, format=IMAGE_FORMAT)

    def save_calib(self, frame, frame_num):
        """ parse and save the calibration data
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
        """
        print("Calibration saving routine")

        Tr_velo_to_cam = []
        calib_context = ''
        camera_calib = []

        waymo_cam_RT=np.array([0,-1,0,0,  0,0,-1,0,   1,0,0,0,    0 ,0 ,0 ,1]).reshape(4,4)

        R0_rect = np.eye(3).flatten().astype(str)

        for camera in frame.context.camera_calibrations:
            tmp=np.array(camera.extrinsic.transform).reshape(4,4)
            tmp=np.linalg.inv(tmp).reshape((16,))
            Tr_velo_to_cam.append(["%e" % i for i in tmp])

        for cam in frame.context.camera_calibrations:
            tmp=np.zeros((3,4))
            tmp[0,0]=cam.intrinsic[0]
            tmp[1,1]=cam.intrinsic[1]
            tmp[0,2]=cam.intrinsic[2]
            tmp[1,2]=cam.intrinsic[3]
            tmp[2,2]=1
            tmp=np.dot(tmp, waymo_cam_RT)
            tmp=list(tmp.reshape(12))
            tmp = ["%e" % i for i in tmp]
            camera_calib.append(tmp)

        for i in range(5):
            calib_context += "P" + str(i) + ": " + " ".join(camera_calib[i]) + '\n'

        calib_context += "R0_rect" + ": " + " ".join(R0_rect) + '\n'

        for i in range(5):
            calib_context += "Tr_velo_to_cam_" + str(i) + ": " + " ".join(Tr_velo_to_cam[i]) + '\n'

        with open(self.CALIB_PATH + '/' + str(frame_num).zfill(INDEX_LENGTH) + '.txt', 'w+') as fp_calib:
            fp_calib.write(calib_context)


    def save_lidar(self, frame, frame_num):
        """ parse and save the lidar data in psd format
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
                """
        print("Lidar points saving routine")
        range_images, range_image_top_pose = self.parse_range_image_and_camera_projection(
            frame)

        points, intensity, nlz_mask = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            range_image_top_pose)

        points_all = np.concatenate(points, axis=0)
        intensity_all = np.concatenate(intensity, axis=0)
        nlz_mask_all = np.concatenate(nlz_mask, axis=0)
        point_cloud = np.column_stack((points_all, intensity_all, nlz_mask_all))

        pc_path = self.LIDAR_PATH + '/' + str(frame_num).zfill(INDEX_LENGTH) + '.bin'
        point_cloud.tofile(pc_path)


    def save_label(self, frame, frame_num):
        """ parse and save the label data in .txt format
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
                """
        print("Labels saving routine")

        # preprocess bounding box data
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                bbox = [label.box.center_x - label.box.length / 2, label.box.center_y - label.box.width / 2,
                        label.box.center_x + label.box.length / 2, label.box.center_y + label.box.width / 2]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        for obj in frame.laser_labels:

            # caculate bounding box
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.__lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break
            if bounding_box == None or name == None:
                continue

            my_type = self.__type_list[obj.type]
            truncated = 0
            occluded = 0
            height = obj.box.height
            width = obj.box.width
            length = obj.box.length
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z
            rotation_y = obj.box.heading
            beta = math.atan2(x, z)
            alpha = (rotation_y + beta - math.pi / 2) % (2 * math.pi)

            # save the labels
            line = my_type + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(round(truncated, 2),
                                                                                   occluded,
                                                                                   round(alpha, 2),
                                                                                   round(bounding_box[0], 2),
                                                                                   round(bounding_box[1], 2),
                                                                                   round(bounding_box[2], 2),
                                                                                   round(bounding_box[3], 2),
                                                                                   round(height, 2),
                                                                                   round(width, 2),
                                                                                   round(length, 2),
                                                                                   round(x, 2),
                                                                                   round(y, 2),
                                                                                   round(z, 2),
                                                                                   round(rotation_y, 2))
            line_all = line[:-1] + ' ' + name + '\n'
            # store the label
            with open(self.LABEL_PATH + "_" + name + '/' + str(frame_num).zfill(INDEX_LENGTH) + '.txt', 'a') as fp_label:
                fp_label.write(line)

        with open(self.LABEL_ALL_PATH + '/' + str(frame_num).zfill(INDEX_LENGTH) + '.txt', 'w+') as fp_label_all:
            fp_label_all.write(line_all)

    def get_file_names(self):
        self.__file_names = []
        for i in os.listdir(self.DATA_PATH):
            if i.split('.')[-1] == 'tfrecord':
                self.__file_names.append(self.DATA_PATH + '/' + i)

    def create_folder(self):
        if not os.path.exists(self.KITTI_PATH):
            os.mkdir(self.KITTI_PATH)
        if not os.path.exists(self.CALIB_PATH):
            os.mkdir(self.CALIB_PATH)
        if not os.path.exists(self.LIDAR_PATH):
            os.mkdir(self.LIDAR_PATH)
        if not os.path.exists(self.LABEL_ALL_PATH):
            os.mkdir(self.LABEL_ALL_PATH)
        for i in range(5):
            if not os.path.exists(self.IMAGE_PATH + "_" + str(i)):
                os.mkdir(self.IMAGE_PATH + "_" + str(i))
            if not os.path.exists(self.LABEL_PATH + "_" + str(i)):
                os.mkdir(self.LABEL_PATH + "_" + str(i))


    def parse_range_image_and_camera_projection(self, frame):
        """Parse range images and camera projections given a frame.
        Args:
           frame: open dataset frame proto
        Returns:
           range_images: A dict of {laser_name,
             [range_image_first_return, range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
              camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
        """
        self.__range_images = {}

        for laser in frame.lasers:
            if len(laser.ri_return1.range_image_compressed) > 0:
                range_image_str_tensor = tf.decode_compressed(
                    laser.ri_return1.range_image_compressed, 'ZLIB')
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                self.__range_images[laser.name] = [ri]

                if laser.name == open_dataset.LaserName.TOP:
                    range_image_top_pose_str_tensor = tf.decode_compressed(
                        laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                    range_image_top_pose = open_dataset.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        bytearray(range_image_top_pose_str_tensor.numpy()))

            if len(laser.ri_return2.range_image_compressed) > 0:
                range_image_str_tensor = tf.decode_compressed(
                    laser.ri_return2.range_image_compressed, 'ZLIB')
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                self.__range_images[laser.name].append(ri)

        return self.__range_images, range_image_top_pose



    def convert_range_image_to_point_cloud(self, frame, range_images, range_image_top_pose, ri_index=0):
        """Convert range images to point cloud.
        Args:
          frame: open dataset frame
           range_images: A dict of {laser_name,
             [range_image_first_return, range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
              camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
          ri_index: 0 for the first return, 1 for the second return.
        Returns:
          points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
          cp_points: {[N, 6]} list of camera projections of length 5
            (number of lidars).
          intensity: {[N, 1]} list of intensity of length 5 (number of lidars).
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)

        points = []

        intensity = []

        nlz_mask = []

        frame_pose = tf.convert_to_tensor(
            np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_top_pose.data),
            range_image_top_pose.shape.dims)

        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])

        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)

        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)

            pixel_pose_local = None
            frame_pose_local = None

            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)

            range_image_mask = range_image_tensor[..., 0] > 0

            nlz_mask_tensor = tf.gather_nd(range_image_tensor[:, :, 3],
                                           tf.where(range_image_mask))

            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            intensity_tensor = tf.gather_nd(range_image_tensor,
                                         tf.where(range_image_mask))

            points.append(points_tensor.numpy())
            intensity.append(intensity_tensor.numpy()[:, 1])
            nlz_mask.append(nlz_mask_tensor.numpy())

        return points, intensity, nlz_mask



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", type=str,
                        required=True, metavar="DIR")

    parser.add_argument("--target_dir", type=str,
                        required=True, metavar="DIR")

    args = parser.parse_args()

    adapter = Adapter(args.source_dir, args.target_dir)
    adapter.cvt()

if __name__ == '__main__':
    main()
