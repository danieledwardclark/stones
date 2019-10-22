# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import KDTree

from .base import Hypothesiser
from ..base import Property
from ..predictor import Predictor
from ..updater import Updater


class FilteredDetectionsHypothesiser(Hypothesiser):
    """Wrapper for Hypothesisers - filters input data

    Wrapper for any type of hypothesiser - filters the 'detections' before
    they are fed into the hypothesiser.
    """

    hypothesiser = Property(
        Hypothesiser, doc="Hypothesiser that is being wrapped.")
    metadata_filter = Property(
        str, doc="Metadata attribute used to filter which detections "
                 "tracks are valid for association.")
    match_missing = Property(
        bool,
        default=True,
        doc="Match detections with missing metadata. Default 'True'.")

    def hypothesise(self, track, detections, *args, **kwargs):
        """
        Parameters
        ==========
        track : :class:`Track`
            A track that contains the target's state
        detections : list of :class:`Detection`
            Retrieved measurements

        Returns
        =======
        : :class:`MultipleHypothesis`
            A list containing the hypotheses between each prediction-detections
            pair.

        Note:   The specific subclass of :class:`SingleHypothesis` returned
                depends on the :class:`Hypothesiser` used.

        """
        track_metadata = track.metadata.get(self.metadata_filter)

        if (track_metadata is None) and self.match_missing:
            match_detections = detections
        else:
            match_metadata = [track_metadata]
            if self.match_missing:
                match_metadata.append(None)

            match_detections = {
                detection for detection in detections
                if detection.metadata.get(
                        self.metadata_filter) in match_metadata}

        return self.hypothesiser.hypothesise(
            track, match_detections, *args, **kwargs)


class DetectionKDTreeFilter(Hypothesiser):
    """Detection kd-tree based filter

    Construct a kd-tree from detections and then use a :class:`~.Predictor` and
    :class:`~.Updater` to get prediction of track in measurement space. This is
    then queried against the kd-tree, and only matching detections are passed
    to the :attr:`hypothesiser`.

    Notes
    -----
    This is only suitable where measurements are in same space as each other
    and at the same timestamp.
    """

    hypothesiser = Property(
        Hypothesiser, doc="Hypothesiser that is being wrapped.")
    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    number_of_neighbours = Property(
        int, default=None,
        doc="Number of neighbours to find. Default `None`, which means all "
            "points within the :attr:`max_distance` are returned.")
    max_distance = Property(
        float, default=np.inf,
        doc="Max distance to return points. Default `inf`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_detections = set()

    def hypothesise(self, track, detections, timestamp, *args, **kwargs):
        # No need for tree here.
        if not detections:
            return self.hypothesiser.hypothesise(
                track, detections, timestamp, *args, **kwargs)

        # Attempt to cache last tree
        if detections is not self._prev_detections:
            self._prev_detections = detections
            self._detections_list = list(detections)
            self._tree = KDTree(
                np.vstack([detection.state_vector[:, 0]
                           for detection in self._detections_list]))

        prediction = self.predictor.predict(track, timestamp)
        measurement_prediction = self.updater.predict_measurement(prediction)

        if self.number_of_neighbours is None:
            indexes = self._tree.query_ball_point(
                measurement_prediction.state_vector.ravel(),
                r=self.max_distance)
        else:
            _, indexes = self._tree.query(
                measurement_prediction.state_vector.ravel(),
                k=self.number_of_neighbours,
                distance_upper_bound=self.max_distance)

        indexes = (index
                   for index in np.atleast_1d(indexes)
                   if index != len(self._detections_list))

        return self.hypothesiser.hypothesise(
            track,
            {self._detections_list[index] for index in indexes},
            timestamp,
            *args, **kwargs)
