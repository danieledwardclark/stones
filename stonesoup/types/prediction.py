# -*- coding: utf-8 -*-
import datetime

from ..base import Property
from .array import CovarianceMatrix
from .base import Type
from .state import (State, GaussianState, ParticleState, SqrtGaussianState,
                    TaggedWeightedGaussianState, ASDGaussianState)


class Prediction(Type):
    """ Prediction type

    This is the base prediction class. """


class MeasurementPrediction(Type):
    """ Prediction type

    This is the base measurement prediction class. """


class StatePrediction(State, Prediction):
    """ StatePrediction type

    Most simple state prediction type, which only has time and a state vector.
    """


class StateMeasurementPrediction(State, MeasurementPrediction):
    """ MeasurementPrediction type

    Most simple measurement prediction type, which only has time and a state
    vector.
    """


class GaussianStatePrediction(Prediction, GaussianState):
    """ GaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class ASDGaussianStatePrediction(Prediction, ASDGaussianState):
    """ ASDGaussianStatePrediction type

    This is a simple ASDGaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """
    act_timestamp = Property(
        datetime.datetime, doc="The timestamp for which "
                               + "the state is predicted")


class SqrtGaussianStatePrediction(Prediction, SqrtGaussianState):
    """ SqrtGaussianStatePrediction type

    This is a Gaussian state prediction object, with the covariance held
    as the square root of the covariance matrix
    """


class WeightedGaussianStatePrediction(Prediction, TaggedWeightedGaussianState):
    """ WeightedGaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution
    with an associated weight.
    """


class TaggedWeightedGaussianStatePrediction(Prediction,
                                            TaggedWeightedGaussianState):
    """ TaggedWeightedGaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution, with an associated
    weight and unique tag.
    """


class GaussianMeasurementPrediction(MeasurementPrediction, GaussianState):
    """ GaussianMeasurementPrediction type

    This is a simple Gaussian measurement prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """

    cross_covar: CovarianceMatrix = Property(
        default=None, doc="The state-measurement cross covariance matrix")

    def __init__(self, state_vector, covar, timestamp=None,
                 cross_covar=None, *args, **kwargs):
        if(cross_covar is not None
           and cross_covar.shape[1] != state_vector.shape[0]):
            raise ValueError("cross_covar should have the same number of "
                             "columns as the number of rows in state_vector")
        super().__init__(state_vector, covar, timestamp,
                         cross_covar, *args, **kwargs)


class ASDGaussianMeasurementPrediction(
                                    MeasurementPrediction, ASDGaussianState):
    """ASD Gaussian Measurement Prediction"""
    cross_covar = Property(CovarianceMatrix,
                           doc="The state-measurement cross covariance matrix",
                           default=None)


class ParticleStatePrediction(Prediction, ParticleState):
    """ParticleStatePrediction type

    This is a simple Particle state prediction object.
    """


class ParticleMeasurementPrediction(MeasurementPrediction, ParticleState):
    """MeasurementStatePrediction type

    This is a simple Particle measurement prediction object.
    """
