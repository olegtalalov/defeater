import os
import numpy as np
import scipy
import torch
import torch.utils.data as tdata
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets


class MatchingCascade(object):
    """
    Creates track matching routine (using deep appearance descriptor and Hungarian algorithm)
    and measures its perfomance via track metrics
    """
    def __init__(self, dataloader, model, metadata_path):
        self.metadata = pd.read_csv(metadata_path)
        self.dataloader = dataloader
        self.model = model

    def get_descriptors(self, source_path=None, dest_path=None):
        """Get descriptors from source path or run model and save descriptors to dest path"""
        assert (source_path is None) ^ (dest_path is None), "Only one path should be provided."
        if source_path:
            self.predictions = np.load(os.path.join(source_path, "pred.npy"))
            self.targets = np.load(os.path.join(source_path, "targets.npy"))
            self.descriptors = np.load(os.path.join(source_path, "deep_cosine_descriptor.npy"))
        else:
            if not os.path.exists(dest_path):
                os.mkdir(dest_path)
            self._model_inference(dest_path)


    def _model_inference(self, dest_path):
        """Run model and save result to dest_path"""
        self.model.to(device)

        deep_cosine_descriptor = []
        predictions = []
        targets = []

        with torch.no_grad():
            for [points, target] in tqdm.tqdm_notebook(self.dataloader):
                points = points.squeeze().transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                pred, _, _, descriptor = self.model(points)
                predictions.append(pred.data.max(1)[1])
                deep_cosine_descriptor.append(descriptor)
                targets.append(target)

        self.descriptors = np.vstack(list(map(lambda x: x.cpu().numpy(),
                                                deep_cosine_descriptor)))
        self.targets = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                                 targets)))
        self.predictions = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                                     predictions)))

        np.save(os.path.join(dest_path, "pred.npy"), np.array(self.predictions))
        np.save(os.path.join(dest_path, "targets.npy"), np.array(self.targets))
        np.save(os.path.join(dest_path, "deep_cosine_descriptor.npy"), np.array(self.descriptors))

    def start_processing(self, thresh=0.3, take_last_n=50, min_pts_num=1, agg_function=np.min, cascading=False, segments_to_proc=None):
        self.thersh = thresh
        self.take_last_n = take_last_n
        self.agg_function = agg_function
        self.track_dict = dict()
        self.all_segment_names = self.metadata['segment_name'].unique()[:segments_to_proc]
        self.metadata = self.metadata[self.metadata['number_of_points'] >= min_pts_num]

        for segment_name, segment_split in self._get_next_segment():
            self._segment_tracks = []
            self._last_track_appearance = []
            for frame_num, frame_split in self._get_next_frame(segment_split):
                if not cascading:
                    self._track_matching(frame_split)
                else:
                    self._cascade_track_matching(frame_split)
            self.track_dict[segment_name] = self._segment_tracks

        self._points_cloud_id_to_track_id()
        self._eval_metrics()

    def _track_matching(self, frame_split):
        if len(self._segment_tracks) == 0:
            self._segment_tracks = frame_split.index.to_numpy().reshape(-1, 1).tolist()
        else:
            cost_matrix = self._compute_cost_matrix(frame_split['point_cloud_id'])
            assert cost_matrix.shape == (len(self._segment_tracks), len(frame_split['point_cloud_id']))

            threshold_matrix = cost_matrix < self.thersh
            track_assignment = scipy.optimize.linear_sum_assignment(cost_matrix)

            for track_id, detection_id in zip(*track_assignment):
                if threshold_matrix[track_id][detection_id]:
                    self._segment_tracks[track_id].append(frame_split['point_cloud_id'].to_numpy()[detection_id])
                else:
                    self._segment_tracks.append([frame_split['point_cloud_id'].to_numpy()[detection_id]])

    def _cascade_track_matching(self, frame_split):
        if len(self._segment_tracks) == 0:
            self._segment_tracks = frame_split.index.to_numpy().reshape(-1, 1).tolist()
            # pair (track_id, frames since last appearance)
            self._last_track_appearance = [(i, 0) for i in range(len(self._segment_tracks))]
        else:
            modified_tracks = set()
            unmatched_detections = np.full(len(frame_split['point_cloud_id']), True)
            # cost matrix of size (Num_of_tracks, Num_of_detections)
            cost_matrix = self._compute_cost_matrix(frame_split['point_cloud_id'])

            assert cost_matrix.shape == (len(self._segment_tracks), len(frame_split['point_cloud_id']))

            threshold_matrix = (cost_matrix < self.thersh)

            # get tracks from newest to oldest and solve optimization problem on bipartite graph
            for appearance_group in itertools.groupby(sorted(self._last_track_appearance, key=lambda x: x[1]), key=lambda x: x[1]):
                track_assignment = scipy.optimize.linear_sum_assignment(
                                                                         cost_matrix[np.ix_(
                                                                                          list(
                                                                                              map(lambda track_id_pair: track_id_pair[0],
                                                                                                    appearance_group[1])
                                                                                              ),
                                                                                          unmatched_detections
                                                                                        )
                                                                                  ]
                                                                        )
                for track_id, detection_id in zip(*track_assignment):
                    if threshold_matrix[track_id][detection_id]:
                        self._segment_tracks[track_id].append(frame_split['point_cloud_id'].to_numpy()[unmatched_detections][detection_id])
                        modified_tracks.add(track_id)

                unmatched_detections[list(track_assignment[1])] = False

            # update add +1 to time since last appearance for each track
            self._last_track_appearance = list(map(lambda x: (x[0], x[1]+1), self._last_track_appearance))
            # set time since last appearance for updated tracks to zero
            for modified_track in modified_tracks:
                self._last_track_appearance[modified_track] = (modified_track, 0)

    def _get_next_segment(self):
        for current_segment in self.all_segment_names:
            yield current_segment, self.metadata.loc[self.metadata['segment_name'] == current_segment]

    def _get_next_frame(self, segment_split):
        for current_frame in segment_split['frame_num'].unique().tolist():
            yield current_frame, segment_split.loc[segment_split['frame_num'] == current_frame]

    def _compute_cost_matrix(self, frame_ids):
        return np.array([self.agg_function(1 - np.dot(np.take(self.descriptors, track[-self.take_last_n:], axis=0),
                                        np.take(self.descriptors, frame_ids, axis=0).T), axis=0) for track in self._segment_tracks])

    def _points_cloud_id_to_track_id(self):
        self.track_dict_ids = dict()
        for segment_name in self.all_segment_names:
            self.track_dict_ids[segment_name] = [self.metadata['track_id'].loc[i].to_numpy()
                                                 for i in self.track_dict[segment_name]]

    def _eval_metrics(self):
        self.track_metrics = { segment_name : [(scipy.stats.mode(track).count[0] / len(track), len(np.unique(track)), len(track))
                                                for track in self.track_dict_ids[segment_name]]
                                                for segment_name in self.all_segment_names}
