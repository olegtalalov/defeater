import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly.graph_objects as go

class TrackScatterPlotter(object):
    """Creates point cloud plot generator for matching cascade results validation."""
    def __init__(self, source_path, track_dict, metadata):
        self.path_to_pc = source_path
        self.track_dict = track_dict
        self.metadata = metadata

    def visualize_track_generator(self, segment_id, track_num):
        self.segment_id = segment_id
        self.track_num = track_num
        for pc_id in self.track_dict[self.segment_id][self.track_num]:
            print(pc_id, self.metadata['track_id'][pc_id])
            track_pc = np.load(os.path.join(self.path_to_pc, f"{pc_id}.npy".zfill(12)))

            fig = go.Figure(data=[go.Scatter3d(x=track_pc[:, 0], y=track_pc[:, 1], z=track_pc[:, 2],
                                       mode='markers')])
            fig.show()
            yield

class PointCloudScatterPlotter(object):
    """Creates simple point cloud plot generator for a given directory."""
    def __init__(self, source_path):
        self.path_to_pc = source_path

    def visualize_pc_generator(self, start_with=0):
        for pc_fn in sorted(os.listdir(self.path_to_pc))[start_with:]:
            print(pc_fn)
            try:
                pc = np.load(os.path.join(self.path_to_pc, pc_fn))
            except:
                pass
            figr = go.Figure(data=[go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
                                       mode='markers')])
            figr.show()
            yield
