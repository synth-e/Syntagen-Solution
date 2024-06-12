from .voc_classifier import VOCClassifier
from synthlab.registry import register, ClassType
from .voc_clipes import VOCCLIPES

register(ClassType.NODE, VOCClassifier)
register(ClassType.NODE, VOCCLIPES)