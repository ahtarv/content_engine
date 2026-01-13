import numpy as np

class UserProfile:
    """"
    Stores a user's intereset and avoidance representatiions
    derived from implicit feedback
    """

    def __init__(self):
        self.read_vecs = []
        self.skip_vecs = []

    def update(self, embedding, dwell_time, scroll_depth):
        """ 
        Update user profile based on interaction strength
        """
        if dwell_time > 30 and scroll_depth > 0.6:
            self.read_vecs.append(embedding)
        else:
            self.skip_vecs.append(embedding)

    @property
    def avoidance_vector(self):
        if not self.skip_vecs:
            return None
        return np.mean(self.skip_vecs, axis=0)