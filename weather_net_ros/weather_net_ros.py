import torch
import sys
from .pcd_de_noising.src.pcd_de_noising import WeatherNet
import h5py
import os
import glob
from typing import List, Dict
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, create_cloud
from std_msgs.msg import Header

DATA_KEYS = ["distance_m_1", "intensity_1"]
LABEL_KEY = "labels_1"
XYZ_KEYS = ["sensorX_1", "sensorY_1", "sensorZ_1"]
COLOR_LABEL_MAPPING = {
    0: [255, 0, 0],
    # 100: [158, 158, 158],
    # 101: [0, 153, 153],
    # 102: [115, 0, 230],
    100: [255, 255, 255],
    101: [255, 255, 255],
    102: [255, 255, 255],
}


class weather_net_ros(Node):
    def __init__(self):
        super().__init__("weather_net_ros_node")

        self.declare_parameter(
            "model_file",
            "/home/toyozoshimada/point-cloud-de-noising/tb_logs/WeatherNet/version_11/checkpoints/epoch=49-step=100.ckpt",
        )
        self.declare_parameter(
            "input_data",
            "/home/toyozoshimada/point-cloud-de-noising/data/test/2018-11-29_104141_Static2-FogB",
        )

        model_file = self.get_parameter("model_file").get_parameter_value().string_value
        input_data = self.get_parameter("input_data").get_parameter_value().string_value

        self.model = WeatherNet.load_from_checkpoint(model_file, num_classes=4)

        input_files = sorted(glob.glob(os.path.join(input_data, "*.hdf5")))
        self.input_sequence: List[List[torch.Tensor]] = []
        for file in input_files:
            with h5py.File(file, "r") as h5_file:
                data = [h5_file[key][()] for key in DATA_KEYS]
                xyz_data = [h5_file[key][()] for key in XYZ_KEYS]
                label = h5_file[LABEL_KEY][()]
            data = tuple(torch.from_numpy(data) for data in data)
            xyz_data = tuple(torch.from_numpy(d) for d in xyz_data)
            data = torch.stack(data)
            xyz_data = torch.stack(xyz_data)
            distance = data[0:1, :, :]  # 1 x 32 x 400
            reflectivity = data[1:, :, :]  # 1 x 32 x 400
            gt = torch.from_numpy(label).long()
            self.input_sequence.append([distance, reflectivity, xyz_data, gt])

        self.input_pcd_idx = 0

        self.__filtered_cloud_pub = self.create_publisher(
            PointCloud2, "pcd/filtered", 10
        )
        self.__timer = self.create_timer(0.1, self.test_inference_timer_cb)

    def get_rgb(self, labels):
        """returns color coding according to input labels"""
        r = g = b = np.zeros_like(labels)
        for label_id, color in COLOR_LABEL_MAPPING.items():
            r = np.where(labels == label_id, color[0] / 255.0, r)
            g = np.where(labels == label_id, color[1] / 255.0, g)
            b = np.where(labels == label_id, color[2] / 255.0, b)
        return r, g, b

    def test_inference_timer_cb(self) -> None:
        print("koko")
        if self.input_pcd_idx >= len(self.input_sequence):
            self.input_pcd_idx = 0
        distance = self.input_sequence[self.input_pcd_idx][0]
        reflectivity = self.input_sequence[self.input_pcd_idx][1]
        xyz_data = self.input_sequence[self.input_pcd_idx][2]
        gt = self.input_sequence[self.input_pcd_idx][3]
        # pred = self.inference(distance, reflectivity)
        pred = self.model.predict(
            distance.unsqueeze(1).to("cuda"), reflectivity.unsqueeze(1).to("cuda")
        )
        # mask = (pred == 2) | (pred == 3)
        # filtered_tensor = xyz_data[:, ~mask.squeeze(0).to("cpu")]

        # points_with_label = torch.cat((xyz_data, pred.to("cpu")), dim=0)
        # points_with_label = torch.cat((xyz_data, gt.reshape(1, 32, 400)), dim=0)
        # print(gt)
        # print(pred[0].shape)

        output_cloud = PointCloud2()
        header = Header()
        header.frame_id = "weather_net"
        header.stamp = self.get_clock().now().to_msg()
        # torch.
        # print(filtered_tensor.tolist())
        # print(filtered_tensor.shape)
        # print(np.shape(filtered_tensor.tolist()))
        # print(points_with_label.view(4, -1).shape)
        fields = [
            PointField(name="x", offset=0, datatype=7, count=1),
            PointField(name="y", offset=4, datatype=7, count=1),
            PointField(name="z", offset=8, datatype=7, count=1),
            PointField(name="r", offset=12, datatype=7, count=1),
            PointField(name="g", offset=16, datatype=7, count=1),
            PointField(name="b", offset=20, datatype=7, count=1),
        ]

        # r, g, b = self.get_rgb(pred.to("cpu").numpy().flatten())
        r, g, b = self.get_rgb(gt.numpy().flatten())

        points = list(
            zip(
                xyz_data[0].numpy().flatten(),
                xyz_data[1].numpy().flatten(),
                xyz_data[2].numpy().flatten(),
                r,
                g,
                b,
            )
        )
        # output_cloud = create_cloud_xyz32(header, points_with_label.t().tolist())
        # output_cloud = create_cloud(header, fields, points_with_label.view(4, -1).t().tolist())
        output_cloud = create_cloud(header, fields, points)
        self.__filtered_cloud_pub.publish(output_cloud)

        self.input_pcd_idx += 1


def main(args=None) -> None:
    rclpy.init(args=args)

    node = weather_net_ros()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
