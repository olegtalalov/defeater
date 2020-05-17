
import os
import numpy as np

class DescriptorsSet(object):
    def __init__(self, max_number, descriptor_size):
        self.max_number = max_number
        self.descriptor_size = descriptor_size
        self.best_descriptors = np.full((max_number, 1 + descriptor_size),
                                    -1, dtype=np.float64)

    def add_batch(self, p_scores, correct, descriptors, score_threshold=0.7, similarity_threshold=0.75):
        [
         self._add_descriptor(p, descriptor, score_threshold, similarity_threshold)
         for p, c, descriptor in zip(p_scores, correct, descriptors) if c
        ]

    def _add_descriptor(self, p, descriptor, score_threshold, similarity_threshold):
        assert len(descriptor) == self.descriptor_size

        if p < score_threshold:
            return None

        empty_slots = np.where(self.best_descriptors[:, 0] < 0)[0]

        with_lesser_score = np.logical_and(self.best_descriptors[:, 0] > 0,
                                           self.best_descriptors[:, 0] < p)

        gated_cosine_similarity = np.dot(self.best_descriptors[:, 1:], descriptor) < similarity_threshold

        worse_descriptors_mask = np.invert(np.logical_and(with_lesser_score, gated_cosine_similarity))

        if not np.all(worse_descriptors_mask):
            drop_index = np.ma.array(self.best_descriptors[:, 0], mask=worse_descriptors_mask).argmin()
            self.best_descriptors[drop_index, 0] = p
            self.best_descriptors[drop_index, 1:] = descriptor

        elif len(empty_slots) and np.all(gated_cosine_similarity[self.best_descriptors[:, 0] > 0]):
            self.best_descriptors[empty_slots[0], 0] = p
            self.best_descriptors[empty_slots[0], 1:] = descriptor

    def get_approx(self, descriptor):
        return self.best_descriptors[np.dot(self.best_descriptors[self.best_descriptors[:, 0] > 0, 1:], descriptor).argmax(), 1:]
