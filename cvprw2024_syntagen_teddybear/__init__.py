from .voc_classifier import VOCClassifier
from synthlab_core.registry import register, ClassType
from .voc_clipes import VOCCLIPES
from .voc_classes import VOCClasses
from . import clipes_utilities
from . import custom_clip
from . import pytorch_grad_cam

register(ClassType.NODE, VOCClassifier)
register(ClassType.NODE, VOCCLIPES)
register(ClassType.NODE, VOCClasses)