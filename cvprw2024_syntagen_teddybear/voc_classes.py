from synthlab_core.node import INode
from synthlab_core.atomic import TextualPrompt, IndexedFile, MaskWrapper
import numpy as np
import structlog
from synthlab_core.utilities.data.label import VOC2012_CATEGORIES

logger = structlog.getLogger(__name__)

class VOCClasses(INode):
    @classmethod
    def in_specs(cls) -> list[tuple[str, type]]:
        return []

    @classmethod
    def out_specs(cls) -> list[tuple[str, type]]:
        return [
            ("prompt", TextualPrompt),
        ]

    def __init__(self, **kwargs):
        self.cat = VOC2012_CATEGORIES[1:]

    def forward(self, *args, **kwargs) -> TextualPrompt:
        return TextualPrompt(labels=self.cat)