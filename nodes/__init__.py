from .voc_classes import VOCClasses
from synthlab.registry import register, ClassType

register(ClassType.NODE, VOCClasses)