from synthlab_core.atomic import TextualPrompt, IndexedFile, ImageWrapper, MaskWrapper
from synthlab_core.node import INode
import numpy as np
import structlog
import torch
from synthlab_core.utilities.data.label import VOC2012_CATEGORIES
from synthlab_core.utilities.data import VOC2012_DATA_CONTEXT

import _clip as clip
from _clipes_utilities.misc import DenseCRF, ClipOutputTarget, scoremap2bbox
from _clipes_utilities.transforms import reshape_transform, img_ms_and_flip_v2
from _pytorch_grad_cam import GradCAM
from _pytorch_grad_cam.utils.image import scale_cam_image
import cv2

logger = structlog.getLogger(__name__)

class_names = [
    'plane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'dining table', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'monitor',
]
                   
aug_class_names = [
    'aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair seat', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
]


background_category = [
    'ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet',
    'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign',
]

mean_bgr = (104.008, 116.669, 122.675)

class VOCCLIPES(INode):
    @classmethod
    def in_specs(cls) -> list[tuple[str, type]]:
        return [
            ("image", ImageWrapper),
            ("target", TextualPrompt),
        ]

    @classmethod
    def out_specs(cls) -> list[tuple[str, type]]:
        return [
            ("prediction", MaskWrapper),
        ]

    def __init__(self, weight_path, **kwargs):
        super().__init__(**kwargs)

        self.model_weight = weight_path  
        self.clip_model, self.clip_preprocess = clip.load(
            self.model_weight, 
            device=self.inference_device if not self.low_resource_mode else self.idle_device
        )

        target_layers = [self.clip_model.visual.transformer.resblocks[-1].ln_1]

        self.cam = GradCAM(
            model=self.clip_model, 
            target_layers=target_layers, 
            reshape_transform=reshape_transform
        )

        self.postprocessor = DenseCRF(
            iter_max=10,
            pos_xy_std=1,
            pos_w=3,
            bi_xy_std=67,
            bi_rgb_std=3,
            bi_w=4,
        )

        self.bg_text_features = self._text_preprocess(background_category)
        self.fg_text_features = self._text_preprocess(aug_class_names)

    @torch.no_grad()
    def _text_preprocess(self, targets: list):
        if isinstance(targets, str):
            targets = [targets]

        zeroshot_weights = []

        try:
            self._ready_to_inference()

            for classname in targets:
                texts = ['a clean origami {}.'.format(classname)] # format with class
                texts = clip.tokenize(texts).to(self.inference_device) # tokenize
                class_embeddings = self.clip_model.encode_text(texts) # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(
                self.idle_device if self.low_resource_mode else self.inference_device
            )
            return zeroshot_weights.t()
        except Exception as err:
            return None
        finally:
            self._completed_inference()
    
    def _ready_to_inference(self):
        if not self.low_resource_mode:
            return
        
        self.clip_model = self.clip_model.to(self.inference_device)

    def _completed_inference(self):
        if not self.low_resource_mode:
            return
        
        self.clip_model = self.clip_model.to(self.idle_device)

    def __call__(self, _img: ImageWrapper, prompt: TextualPrompt, *args, **kwargs) -> TextualPrompt:
        labels = prompt.labels
        img = _img.pil
        
        label_list = []
        label_id_list = []

        for obj in labels:
            obj = aug_class_names[class_names.index(obj)]

            if obj not in label_list:
                label_list.append(obj)
                label_id_list.append(aug_class_names.index(obj))
        
        if len(label_list) == 0:
            return MaskWrapper(np.zeros(_img.size(), dtype=np.uint8))
        
        ori_width, ori_height = img.size
        ms_imgs = img_ms_and_flip_v2(img, ori_height, ori_width, scales=[1.0])
        ms_imgs = [ms_imgs[0]]
        highres_cam_all_scales = []
        refined_cam_all_scales = []

        for image in ms_imgs:
            image = image.unsqueeze(0)
            h, w = image.shape[-2], image.shape[-1]
            image = image.to(self.inference_device)
            image_features, attn_weight_list = self.clip_model.encode_image(image, h, w)

            highres_cam_to_save = []
            refined_cam_to_save = []
            keys = []

            bg_features_temp = self.bg_text_features.to(self.inference_device)  # [bg_id_for_each_image[im_idx]].to(device_id)
            fg_features_temp = self.fg_text_features[label_id_list].to(self.inference_device)
            text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
            input_tensor = [image_features, text_features_temp.to(self.inference_device), h, w]

            for idx, label in enumerate(label_list):
                keys.append(aug_class_names.index(label))
                targets = [ClipOutputTarget(label_list.index(label))]

                #torch.cuda.empty_cache()
                grayscale_cam, _ , attn_weight_last = self.cam(
                    input_tensor=input_tensor,
                    targets=targets,
                    target_size=None
                )  # (ori_width, ori_height))

                grayscale_cam = grayscale_cam[0, :]

                grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))
                highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

                if idx == 0:
                    attn_weight_list.append(attn_weight_last)
                    attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
                    attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                    attn_weight = torch.mean(attn_weight, dim=0)
                    attn_weight = attn_weight[0].cpu().detach()
                attn_weight = attn_weight.float()

                box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
                aff_mask = torch.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1]))
                for i_ in range(cnt):
                    x0_, y0_, x1_, y1_ = box[i_]
                    aff_mask[y0_:y1_, x0_:x1_] = 1

                aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
                aff_mat = attn_weight

                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

                for _ in range(2):
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
                trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

                for _ in range(1):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                trans_mat = trans_mat * aff_mask

                cam_to_refine = torch.FloatTensor(grayscale_cam)
                cam_to_refine = cam_to_refine.view(-1,1)

                # (n,n) * (n,1)->(n,1)
                cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16)
                cam_refined = cam_refined.cpu().numpy().astype(np.float32)
                cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
                refined_cam_to_save.append(torch.tensor(cam_refined_highres))

            keys = torch.tensor(keys)
            #cam_all_scales.append(torch.stack(cam_to_save,dim=0))
            highres_cam_all_scales.append(torch.stack(highres_cam_to_save,dim=0))
            refined_cam_all_scales.append(torch.stack(refined_cam_to_save,dim=0))


        # cam_all_scales = cam_all_scales[0]
        highres_cam_all_scales = highres_cam_all_scales[0]
        refined_cam_all_scales = refined_cam_all_scales[0]

        cv2img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR).astype(np.float32)

        cv2img -= mean_bgr
        cv2img = cv2img.transpose(2, 0, 1)
        cams = refined_cam_all_scales.cpu().numpy().astype(np.float16)
        bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)
        cams = np.concatenate((bg_score, cams), axis=0)
        prob = cams

        cv2img = cv2img.astype(np.uint8).transpose(1, 2, 0)
        prob = self.postprocessor(cv2img, prob)
       
        label = np.argmax(prob, axis=0)
        keys = np.pad(keys.numpy() + 1, (1, 0), mode='constant')
        label = keys[label]

        confidence = np.max(prob, axis=0)
        label[confidence < 0.95] = 255

        return MaskWrapper(label.astype(np.uint8), labels=VOC2012_DATA_CONTEXT.id2label)