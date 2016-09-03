import warnings
warnings.warn("ClusterCentroids has been moved to "
              "clustering.ClusterCentroids "
              "in 0.2 and will be removed in 0.4", DeprecationWarning)

from .clustering import ClusterCentroids
