import numpy as np


class UserProfile:
    """
    Stores a user's interest and avoidance representations
    derived from implicit feedback.
    """

    def __init__(self):
        self.read_vecs = []
        self.skip_vecs = []

    def update(self, embedding, dwell_time, scroll_depth):
        if dwell_time > 30 and scroll_depth > 0.6:
            self.read_vecs.append(embedding)
        else:
            self.skip_vecs.append(embedding)

    @property
    def interest_vector(self):
        if len(self.read_vecs) == 0:
            return None
        return np.mean(self.read_vecs, axis=0)

    @property
    def avoidance_vector(self):
        if len(self.skip_vecs) == 0:
            return None
        return np.mean(self.skip_vecs, axis=0)
