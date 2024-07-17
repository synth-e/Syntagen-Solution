from synthlab.module.cv.classification.base import IImageClassifier
from synthlab.common.atomic import TextualPrompt, IndexedFile, ImageWrapper
import numpy as np
import structlog
import clip
import torch
import gdown
import os
from synthlab.utilities.data.label import VOC2012_CATEGORIES
import traceback

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

class VOCClassifier(IImageClassifier):
    @classmethod
    def in_specs(cls) -> list[tuple[str, type]]:
        return [
            ("image", ImageWrapper),
        ]

    @classmethod
    def out_specs(cls) -> list[tuple[str, type]]:
        return [
            ("predictions", TextualPrompt),
        ]

    def __init__(self, gdrive_id, **kwargs):
        super().__init__(**kwargs)

        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", 
            device=self.inference_device if not self.low_resource_mode else self.idle_device
        )

        self.classifier = VOCMultiLabelClassifier(self.clip_model.visual)

        os.makedirs('.tmp', exist_ok=True)
        if not os.path.exists(f'.tmp/{gdrive_id}.pth'):
            gdown.download(id=gdrive_id, output=f'.tmp/{gdrive_id}.pth')

        assert os.path.exists(f'.tmp/{gdrive_id}.pth'), f"Model file {gdrive_id} not found"

        self.classifier.classifier = torch.load(
            f'.tmp/{gdrive_id}.pth', 
            map_location=self.inference_device if not self.low_resource_mode else self.idle_device
        )

        self.voc_class_names = VOC2012_CATEGORIES

        self.linker = {
            v: self.voc_class_names.index(k)
            for k, v in VOCMultiLabelClassifier.labels().items()
        }
        
        if not self.low_resource_mode:
            self.classifier.to(self.inference_device)
    
    def _ready_to_inference(self):
        if not self.low_resource_mode:
            return

        self.classifier = self.classifier.to(self.inference_device)

    def _completed_inference(self):
        if not self.low_resource_mode:
            return

        self.classifier = self.classifier.to(self.idle_device)

    @torch.no_grad()
    def __call__(self, img: ImageWrapper, *args, **kwargs) -> TextualPrompt:
        
        try:
            self._ready_to_inference()
            inp = self.clip_preprocess(img.pil).unsqueeze(0).to(self.inference_device)
            logits = self.classifier(inp)
            act = torch.nn.functional.sigmoid(logits).squeeze(0).cpu().numpy() > 0.5
            return TextualPrompt(
                labels=[self.voc_class_names[self.linker[i]] for i, v in enumerate(act) if v]
            )
        except Exception as err:
            traceback.print_exc()

            return TextualPrompt(labels=[])

        finally:
            self._completed_inference()