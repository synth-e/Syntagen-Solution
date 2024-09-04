from synthlab_core.atomic import TextualPrompt, IndexedFile, ImageWrapper
from synthlab_core.node import INode
import numpy as np
import structlog
import clip
import torch
from synthlab_core.utilities.data.label import VOC2012_CATEGORIES

logger = structlog.getLogger(__name__)

class VOCMultiLabelClassifier(torch.nn.Module):
    def __init__(self, feature_extractor, num_classes = 20):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.feature_extractor.output_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.num_classes)
        )
        
    def forward(self, imgs):
        features = self.feature_extractor(
            imgs.to(dtype = self.feature_extractor.conv1.weight.dtype)
        ).to(dtype = self.classifier[0].weight.dtype)
        return self.classifier(features)
    
    @classmethod
    def labels(cls):
        return {'cat': 0, 'monitor': 1, 'car': 2, 'bus': 3, 'bottle': 4, 'bird': 5, 'cow': 6, 'sheep': 7, 'motorbike': 8, 'sofa': 9, 'plane': 10, 'bicycle': 11, 'chair': 12, 'boat': 13, 'potted plant': 14, 'horse': 15, 'train': 16, 'person': 17, 'dining table': 18, 'dog': 19}

class VOCClassifier(INode):
    @classmethod
    def in_specs(cls) -> list[tuple[str, type]]:
        return [
            ("image", ImageWrapper),
        ]

    @classmethod
    def out_specs(cls) -> list[tuple[str, type]]:
        return [
            ("prompt", TextualPrompt),
        ]

    def __init__(self, weight_path, **kwargs):
        super().__init__(**kwargs)

        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", 
            device=self.inference_device 
            if not self.switch_device else self.idle_device
        )
        
        self.classifier = VOCMultiLabelClassifier(self.clip_model.visual)
        
        if weight_path is not None:
            self.classifier.load_state_dict(
                torch.load(
                    weight_path, 
                    map_location=self.inference_device 
                    if not self.switch_device else self.idle_device
                )
            )

        self.voc_class_names = VOC2012_CATEGORIES

        self.linker = {
            v: self.voc_class_names.index(k)
            for k, v in VOCMultiLabelClassifier.labels().items()
        }
        
        if not self.switch_device:
            self.classifier.to(self.inference_device)
    
    @torch.no_grad()
    def forward(self, img: ImageWrapper) -> TextualPrompt:
        inp = self.clip_preprocess(img.pil).unsqueeze(0).to(self.inference_device)
        logits = self.classifier(inp)
        act = torch.nn.functional.sigmoid(logits).squeeze(0).cpu().numpy() > 0.5
        return TextualPrompt(
            labels=[self.voc_class_names[self.linker[i]] 
                    for i, v in enumerate(act) if v]
        )