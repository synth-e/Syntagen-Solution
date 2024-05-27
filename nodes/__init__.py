from .voc_classifier import VOCClassifier
from synthlab.registry import register, ClassType

register(ClassType.NODE, VOCClassifier)