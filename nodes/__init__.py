from .voc_classifier import VOCClasses
from synthlab.registry import register, ClassType

register(ClassType.NODE, VOCClasses)