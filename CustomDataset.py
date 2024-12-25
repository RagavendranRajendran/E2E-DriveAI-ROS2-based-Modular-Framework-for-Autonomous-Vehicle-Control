import torch, os, json, cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.stats import mode
import open3d as o3d
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, masterinfo_path, config_path):

        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

        self.past = []
        self.future = []

        self.req_in_freq = self.config["req_in_freq"]
        self.req_op_freq = self.config["req_op_freq"]
        self.observation_window = self.config["observation_window"]
        self.control_window = self.config["control_window"]

        self.past, self.future = self.sample_data(masterinfo_path, self.req_in_freq, self.req_op_freq, self.observation_window, self.control_window)

    def __len__(self):
        return len(self.past)

    def __getitem__(self, idx):
        """
        Retrieve an item by index. also extract pcd and img data from the path in variable: Past
        """

        num_entries_per_sample = self.observation_window * self.req_in_freq
        # Image shape (width, height)
        w, h = 2088, 1088

        # Initialize img as a 4D NumPy array for efficient storage
        img = np.empty((num_entries_per_sample, h, w, 3), dtype=np.float32)

        # For pcd, initialize as a list (since dimensions vary)
        pcd = [None] * num_entries_per_sample

        img_paths = [str(self.past[idx][j][1]) for j in range(num_entries_per_sample)]
        pcd_paths = [str(self.past[idx][j][0]) for j in range(num_entries_per_sample)]

        img = self.getimg(img_paths)  # Assign directly to pre-allocated array
        pcd = self.getpcd(pcd_paths)  # Assign to pre-allocated list

        # Convert data to appropriate types
        img = torch.tensor(img, dtype=torch.float32)  # Convert images to a Torch tensor

        # Prepare x dictionary
        x = {
            "Image": img,
            "PCD": pcd
        }
        # Prepare y dictionary (assuming speed and steer data are stored in self.future[idx])
        y = {
            "Speed": torch.tensor(self.future[idx][:, 0], dtype=torch.float32),  # Speed column
            "Steer": torch.tensor(self.future[idx][:, 1], dtype=torch.float32)  # Steering column
        }
        return x, y

    def estimate_frequency(self, df, column_name):
        timestamps = df[column_name].dropna().values  # Drop NaN values if any
        time_deltas = np.diff(timestamps)  # Compute differences between consecutive timestamps

        # Filter out anomalies using median absolute deviation (MAD)
        median_delta = np.median(time_deltas)
        mad = np.median(np.abs(time_deltas - median_delta))
        threshold = 3 * mad  # Adjust as needed for sensitivity
        filtered_deltas = time_deltas[np.abs(time_deltas - median_delta) <= threshold]
        frequency = 1 / np.median(filtered_deltas)  # Use median of filtered deltas for robust estimation
        return frequency

    def sample_data(self, masterinfo_path, req_in_freq, req_op_freq, observation_window, control_window):

        masterinfo = pd.read_csv(masterinfo_path)

        # Iterate through each row in the masterinfo file
        for index, row in masterinfo.iterrows():

            rosbag_prefix = row['ROS bags']  # To save sample names with indices
            train_test = row['TrainTest'] #To know if this scenario is for training or test
            in_csv_path = row['IN']  # Path to IN CSV
            op_csv_path = row['OP']  # Path to OP CSV
            in_sample_path = f"{os.path.dirname(in_csv_path)}/IN samples"
            op_sample_path = f"{os.path.dirname(op_csv_path)}/OP samples"
            os.makedirs(in_sample_path, exist_ok=True)
            os.makedirs(op_sample_path, exist_ok=True)

            # Read IN and OP CSV files
            in_df = pd.read_csv(in_csv_path)
            op_df = pd.read_csv(op_csv_path)

            est_in_freq = self.estimate_frequency(in_df, 'Timestamp')
            est_op_freq = self.estimate_frequency(op_df, 'Timestamp')

            if req_in_freq > round(est_in_freq):
                print(
                    f"Expected frequency of past data: {req_in_freq} is more than actual frequency of data: {est_in_freq}")
                break

            if req_op_freq > round(est_op_freq):
                print(
                    f"Expected frequency of future data: {req_op_freq} is more than actual frequency of data: {est_op_freq}")
                break

            # Calculate step sizes for sliding windows
            in_step = int(round(est_in_freq / req_in_freq)) if est_in_freq and req_in_freq else 1
            op_step = int(round(est_op_freq / req_op_freq)) if est_op_freq and req_op_freq else 1

            in_required_rows = int(observation_window * req_in_freq)
            op_required_rows = int(control_window * req_op_freq)

            sample_index = 1
            start_row = 0

            while start_row + in_required_rows < len(in_df):

                in_sample_indices = range(start_row, start_row + in_required_rows * in_step, in_step)

                if max(in_sample_indices, default=0) >= len(in_df) or len(
                        in_df.iloc[in_sample_indices]) < in_required_rows:
                    print(f"Done sampling IN after {sample_index - 1} samples. Stopping...")
                    break

                in_sample = in_df.iloc[in_sample_indices] if max(in_sample_indices, default=0) < len(
                    in_df) else in_df.iloc[
                                start_row:]

                # Find the closest timestamp in OP CSV for the last timestamp in IN sample
                last_timestamp = in_sample['Timestamp'].iloc[-1]
                op_closest_idx = op_df[op_df['Timestamp'] > last_timestamp]['Timestamp'].idxmin()

                # Stops the loop if it cannot find the closest op timestamp value beyond the given timestamp
                if pd.isna(op_closest_idx):
                    print(f"Done sampling OP after {sample_index - 1} samples. No valid OP timestamp found.")
                    break

                # Get the next required number of rows from OP with step
                op_sample_indices = range(op_closest_idx, op_closest_idx + op_required_rows * op_step, op_step)

                if max(op_sample_indices, default=0) >= len(op_df) or len(
                        op_df.iloc[op_sample_indices]) < op_required_rows:
                    print(f"Done sampling OP after {sample_index - 1} samples. Stopping...")
                    break

                op_sample = op_df.iloc[op_sample_indices] if max(op_sample_indices, default=0) < len(
                    op_df) else op_df.iloc[
                                op_closest_idx:]

                in_sample.insert(3, 'ROS_bag', rosbag_prefix)  # Add 'ROS_bag' as the 3rd column
                in_sample.insert(4, 'TrainTest', str(train_test))  # Add 'TrainTest' as the 4th column

                # Print or save the sampled data
                '''print(f"Sample {sample_index}: IN ({len(in_sample)}) rows, OP ({len(op_sample)}) rows.")
                in_sample.to_csv(f"{in_sample_path}/{rosbag_prefix}_IN_sample_{sample_index}.csv", index=False)
                op_sample.to_csv(f"{op_sample_path}/{rosbag_prefix}_OP_sample_{sample_index}.csv", index=False)'''

                self.past.append(in_sample.drop(columns=['Timestamp']).to_numpy())
                self.future.append(op_sample.drop(columns=['Timestamp']).to_numpy())

                # Move to the next batch of 50 rows in IN
                start_row += in_step
                sample_index += 1

        return self.past, self.future

    def getimg(self, img_paths):
        """
        Load the batch of images from the given path.
        """
        img_batch = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_data = img / 255.0  # Normalize image data
            img_batch.append(img_data)

        # Convert to a NumPy array for efficient storage
        img_batch = np.array(img_batch, dtype=np.float32)
        return img_batch

    def getpcd(self, pcd_paths):

        pcd_batch = []
        for pcd_path in pcd_paths:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            if pcd.colors:
                colors = np.asarray(pcd.colors)
                pcd_data = np.hstack((points, colors))  # Combine points and colors
            else:
                pcd_data = points
            pcd_batch.append(pcd_data)
        return pcd_batch