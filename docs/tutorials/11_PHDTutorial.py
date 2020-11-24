#!/usr/bin/env python
# coding: utf-8

"""
=============================================================
11 - Tracking with the Gaussian Mixture Probaiblity Hypothesis Density (GM-PHD) Filter
=============================================================
In previous tutorials, the JPDA and MHT were used to in a multi-target trackign scenario. Here we use a 
Gaussian Mixture (GM) implementation of the Probaiblity Hypothesis Density (PHD) filter to track an 
unknown number of targets 
"""

# %%
# Process
# -------
# This notebook, as with the previous, proceeds according to the following steps:
#
# 1. Create the simulation
#
#   * Initialise the 'playing field'
#   * Choose number of targets and initial states
#   * Create some transition models
#   * Create some sensor models
#
# 2. Initialise the tracker components
#
#   * Initialise predictors
#   * Initialise updaters
#   * Initialise data associations, hypothesisers
#   * Create the tracker
#
# 3. Run the tracker
#
#   * Plot the output
#

# %%
# Create the simulation
# -----------------------

# %%
# Separate out the imports
import numpy as np
import datetime

# %%
# Initialise ground truth
# ^^^^^^^^^^^^^^^^^^^^^^^
# Here are some configurable parameters associated with the ground truth, e.g. defining where
# tracks are born and at what rate, death probability. This follows similar logic to the code
# in previous tutorial section :ref:`auto_tutorials/09_Initiators_&_Deleters:Simulating Multiple
# Targets`.
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
initial_state_mean = StateVector([[0], [0], [0], [0]])
initial_state_covariance = CovarianceMatrix(np.diag([4, 0.5, 4, 0.5]))
timestep_size = datetime.timedelta(seconds=5)
number_of_steps = 20
birth_rate = 0.2
death_probability = 0.001
initial_state = GaussianState(initial_state_mean, initial_state_covariance)

# %%
# Create the transition model - default set to 2d nearly-constant velocity with small (0.05)
# variance.
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])

## %
# Put this all together in a multi-target simulator.
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    timestep=timestep_size,
    number_steps=number_of_steps,
    birth_rate=birth_rate,
    death_probability=death_probability
)

# %%
# Initialise the measurement models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The simulated ground truth will then be passed to a simple detection simulator. This again has a
# number of configurable parameters, e.g. where clutter is generated and at what rate, and
# detection probability. This implements similar logic to the code in the previous tutorial section
# :ref:`auto_tutorials/09_Initiators_&_Deleters:Generate Detections and Clutter`.
from stonesoup.simulator.simple import SimpleDetectionSimulator
from stonesoup.models.measurement.linear import LinearGaussian

# initialise the measurement model
measurement_model_covariance = np.diag([0.25, 0.25])
measurement_model = LinearGaussian(4, [0, 2], measurement_model_covariance)

# probability of detection
probability_detection = 0.9

# clutter will be generated uniformly in this are around the target
clutter_area = np.array([[-1, 1], [-1, 1]])*30
clutter_rate = 1

# %%
# The detection simulator
detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    detection_probability=probability_detection,
    meas_range=clutter_area,
    clutter_rate=clutter_rate
)

# %%
# Create the tracker components
# -----------------------------
# In this example a Kalman filter is used.
#

# %%
# Predictor
# ^^^^^^^^^
# Initialise the predictor using the same transition model as generated the ground truth. Note you
# don't have to use the same model.
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

# %%
# Updater
# ^^^^^^^
# Initialise the updater using the same measurement model as generated the simulated detections.
# Note, again, you don't have to use the same model (noise covariance).
from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# Data associator
# ^^^^^^^^^^^^^^^
# Initialise a hypothesiser which will rank predicted measurement - measurement pairs according to
# some measure.
# Initialise a Mahalanobis distance measure to facilitate this ranking.
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
base_hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=16)

# %%
# Initialise the multi-target hypothesiser with the hypothesiser.
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
hypothesiser = GaussianMixtureHypothesiser(hypothesiser=base_hypothesiser, order_by_detection=True)


# %%
# Reducer
# ^^^^^^^^^^^^^^^^^^^^^
# And a Gaussian Mixture reducer to remove low weighted components (pruning) and to merge similar (overlapping) components (merging). This is done to reduce computational complexity.
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
merge_threshold = 4
prune_threshold = 1e-3
reducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                 merge_threshold=merge_threshold)

# %%
# We will also need to create a "birth" component. This models the expected number of new targets at each time step. This is acheived by having a large Gaussian component, which covers the entire state space, and is therefore associated with every measurement.
from stonesoup.types.state import TaggedWeightedGaussianState

birth_component = TaggedWeightedGaussianState(StateVector([[0], [0], [0], [0]]),CovarianceMatrix(np.diag([1, 1, 1, 1])),weight=1,tag="birth",timestamp=datetime.datetime.now())

# %%
# Run the Tracker
# ---------------
# With all the components in place, we'll now construct the tracker with a multi target tracker. 
from stonesoup.updater.pointprocess import PHDUpdater
from stonesoup.tracker.pointprocess import PointProcessMultiTargetTracker

phd_updater = PHDUpdater(updater=updater, prob_detection=0.9)

tracker = PointProcessMultiTargetTracker(
    detector=detection_sim,
    updater=phd_updater,
    hypothesiser=hypothesiser,
    reducer=reducer,
    birth_component=birth_component
    )
# %%
# Plot the outputs
# ^^^^^^^^^^^^^^^^
# We plot the output using a Stone Soup :class:`MetricGenerator` which does plots (in this instance
# :class:`TwoDPlotter`. This will produce plots equivalent to that seen in previous tutorials.
groundtruth = set()
detections = set()
tracks = set()
for time, ctracks in tracker:
    groundtruth.update(groundtruth_sim.groundtruth_paths)
    detections.update(detection_sim.detections)
    tracks.update(ctracks)

from stonesoup.metricgenerator.plotter import TwoDPlotter

plotter = TwoDPlotter(track_indices=[0, 2], gtruth_indices=[0, 2], detection_indices=[0, 1])
fig = plotter.plot_tracks_truth_detections(tracks, groundtruth, detections).value

ax = fig.axes[0]
ax.set_xlim([-30, 30])
_ = ax.set_ylim([-30, 30])

# %%
