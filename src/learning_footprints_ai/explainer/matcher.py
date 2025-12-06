from typing import Any, Dict

import numpy as np

class FootprintMatcher:
    """
    Given a new activation (compressed), find similar past footprints.
    For Week 1 this is only a placeholder.
    """
    def __init__(self):
        self.footprints = None  # will hold a matrix of [N_footprints, dim]

    def load_footprints(self, path: str):
        # TODO: implement in Week 4 (e.g., np.load(path))
        pass

    def match(self, activation: np.ndarray) -> Dict[str, Any]:
        """
        Return information about the closest footprints.
        """
        # TODO: implement cosine / Euclidean similarity in Week 5
        return {}
