import rclpy, csv, os, cv2
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d


bag_path = "/home/t14-3/Documents/readtest/rosbags/rosbag2_2024_10_22-18_22_35/rosbag2_2024_10_22-18_22_35_0.db3"
bag_name = os.path.splitext(os.path.basename(bag_path))[0]
OP_output_path = "/home/t14-3/Documents/readtest/" + bag_name + "_OP_data.csv"
IN_output_path = "/home/t14-3/Documents/readtest/" + bag_name + "_IN_data.csv"

def synchronize_arrays(data1, data2): # Synchronized to timestamp of data1

	data1 = data1[data1[:, 0].argsort()] # sorting timestamps of both the data
	data2 = data2[data2[:, 0].argsort()]

	timestamps1, values1 = data1[:, 0].astype(float), data1[:, 1:]
	timestamps2, values2 = data2[:, 0].astype(float), data2[:, 1:]

	nearest_indices = np.abs(timestamps1[:, None] - timestamps2).argmin(axis=1)

	synchronized_values2 = values2[nearest_indices]

	synchronized_data = np.hstack((data1, synchronized_values2))

	return synchronized_data

def extract_pointcloud_data(msg):
    """Extract point data from a sensor_msgs.msg.PointCloud2 message.
    Args: msg (sensor_msgs.msg.PointCloud2): The PointCloud2 message.
    Returns: list: A list of points, where each point is a tuple (x, y, z, intensity, [other fields if present])."""

    points = list(point_cloud2.read_points(msg, field_names=["x", "y", "z", "intensity"], skip_nans=True))
    return points


def save_as_pcd(points, output_file):
    # Convert points to Open3D format

    points_np = np.array(points)
    points_np = np.vstack(points_np)
    points_np = np.squeeze(np.array(points_np.tolist()))
    intensity = [p1[3] for p1 in points_np]
    intensity = intensity / np.max(intensity)
    colors = np.tile(intensity[:, None], (1, 3))
    points_np = [p[:3] for p in points_np]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_file, point_cloud)


def dataextractor(bag_path, IN_output_path):

	IN_directory = os.path.dirname(IN_output_path)
	pcd_output_folder = IN_directory + "/pcdout/"
	img_output_folder = IN_directory + "/imgout/"
	os.makedirs(pcd_output_folder, exist_ok=True)
	os.makedirs(img_output_folder, exist_ok=True)

	rclpy.init()
	storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
	converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
	reader = SequentialReader()

	try:
		reader.open(storage_options, converter_options)
	except RuntimeError as e:
		print(f"Failed to open bag file. Error: {e}")
		return

	topic_types = reader.get_all_topics_and_types()
	type_dict = {topic.name: topic.type for topic in topic_types}

	topics_to_read = ["/SpeedoInfo", "/SteerWheelInfo", "/velodyne_points", "/camera/image_raw"]
	valid_topics = [topic for topic in topics_to_read if topic in type_dict]

	if not valid_topics:
		print("No valid topics found in the bag file matching the given list.")
		return

	print(f"Reading messages from topics: {valid_topics}")

	speedo_data, steerwheel_data = [], []

	pcd_filenames, img_filenames = [], []

	while reader.has_next():
		topic, data, timestamp = reader.read_next()

		if topic in valid_topics:
			msg_type = get_message(type_dict[topic])
			msg = deserialize_message(data, msg_type)

			if topic == "/SpeedoInfo":
				speedo_timestamp = f"{msg.timestamp_speedo_in_dec.sec}.{msg.timestamp_speedo_in_dec.nanosec}"
				speedo_data.append([float(speedo_timestamp), msg.speedo_in_dec])

			elif topic == "/SteerWheelInfo":
				steer_timestamp = f"{msg.timestamp_sw_pos_in_dec.sec}.{msg.timestamp_sw_pos_in_dec.nanosec}"
				steerwheel_data.append([float(steer_timestamp), msg.sw_pos_in_dec])

			elif topic == "/velodyne_points":

				msg_type = get_message(type_dict[topic])
				msg = deserialize_message(data, msg_type)
				points = extract_pointcloud_data(msg)

				velodyne_timestamp = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
				pcd_output_path = pcd_output_folder + bag_name + "_pcd_" + velodyne_timestamp + ".pcd"
				save_as_pcd(points, pcd_output_path)  # Save the point cloud to a file
				pcd_filenames.append([float(velodyne_timestamp), pcd_output_path])

			elif topic == "/camera/image_raw":

				msg_type = get_message(type_dict[topic])
				msg = deserialize_message(data, msg_type)
				img_timestamp = timestamp / 1e9  # Camera msg doesn't have a header information for timestamps
				bridge = CvBridge()
				cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
				image_filepath = img_output_folder + bag_name + "_img_" + str(img_timestamp) + ".jpg"

				cv2.imwrite(image_filepath, cv_image)
				img_filenames.append([float(img_timestamp), image_filepath])

	rclpy.shutdown()
	IN_data = synchronize_arrays(np.array(pcd_filenames), np.array(img_filenames))
	OP_data = synchronize_arrays(np.array(speedo_data), np.array(steerwheel_data))

	return IN_data, OP_data

IN_data, OP_data = dataextractor(bag_path, IN_output_path)

# Write the synchronized Output data to a CSV file
with open(OP_output_path, mode='w', newline='') as csvfile:
	csv_writer = csv.writer(csvfile)
	csv_writer.writerow(["Timestamp", "SpeedoInfo", "SteerWheelInfo"])

	for row in OP_data:
		csv_writer.writerow(row)

print(f"Output (Speedo, Steer data) written to {OP_output_path}")

# Write the synchronized Input data to a CSV file
with open(IN_output_path, mode='w', newline='') as csvfile:
	csv_writer = csv.writer(csvfile)
	csv_writer.writerow(["Timestamp", "PCD filename", "IMG filename"])

	for row in IN_data:
		csv_writer.writerow(row)

print(f"Input (LiDAR, Image data) written to {IN_output_path}")
