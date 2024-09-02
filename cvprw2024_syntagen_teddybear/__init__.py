from .voc_classifier import VOCClassifier
from synthlab_core.registry import register, ClassType
from .voc_clipes import VOCCLIPES
from .voc_classes import VOCClasses
from . import _clipes_utilities
from . import _clip
from . import _pytorch_grad_cam

register(ClassType.NODE, VOCClassifier)
register(ClassType.NODE, VOCCLIPES)
register(ClassType.NODE, VOCClasses)