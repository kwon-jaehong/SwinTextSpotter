import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from detectron2.layers import Conv2d, get_norm
from .MaskEncoding import PCAMaskEncoding
from detectron2.modeling.backbone import PatchEmbed

__all__ = ["SWINTS"]


## 이미지 특징 추출기 -> 평균값
## 이것도 proposal 피쳐 초기화임
class ImgFeatExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.img_feat_layer = nn.AdaptiveAvgPool2d(1)
        self.cfg = cfg

    def forward(self, features):
        for i, f in enumerate(features):
            if i == 0:
                # features[0]
                # torch.Size([1, 256, 192, 240])
                x = torch.mean(torch.mean(f, -1), -1) #self.img_feat_layer(f)
                # x.shape
                # torch.Size([1, 256])
                
            else:
                x_p = torch.mean(torch.mean(f, -1), -1) #self.img_feat_layer(f)
                x = x + x_p
        # x = 
        # tensor([[ 3.9198e-01, -8.7791e-01, -2.5660e+00, -1.7946e+00,  3.0572e-01,
                            # .........
        #  -7.2503e-02, -8.4723e-02,  2.3512e+00, -3.7191e-01, -5.2184e-01,
        #   6.2713e-02]], device='cuda:0', grad_fn=<AddBackward0>)
        img_feats = x.squeeze(-1).squeeze(-1).unsqueeze(1).repeat(1, self.cfg.MODEL.SWINTS.NUM_PROPOSALS, 1,)
    #     tensor([[[ 0.3920, -0.8779, -2.5660,  ..., -0.3719, -0.5218,  0.0627],
    #      [ 0.3920, -0.8779, -2.5660,  ..., -0.3719, -0.5218,  0.0627],
    #      [ 0.3920, -0.8779, -2.5660,  ..., -0.3719, -0.5218,  0.0627],
    #      ...,
    #      [ 0.3920, -0.8779, -2.5660,  ..., -0.3719, -0.5218,  0.0627],
    #      [ 0.3920, -0.8779, -2.5660,  ..., -0.3719, -0.5218,  0.0627],
    #      [ 0.3920, -0.8779, -2.5660,  ..., -0.3719, -0.5218,  0.0627]]],
    #    device='cuda:0', grad_fn=<RepeatBackward>)
        
        del x_p
        del x
        
        return img_feats


@META_ARCH_REGISTRY.register()
class SWINTS(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # cfg.MODEL.ROI_HEADS.IN_FEATURES ['p2', 'p3', 'p4', 'p5']
        
        self.num_classes = cfg.MODEL.SWINTS.NUM_CLASSES
        # cfg.MODEL.SWINTS.NUM_CLASSES 2
        
        
        self.num_proposals = cfg.MODEL.SWINTS.NUM_PROPOSALS
        # cfg.MODEL.SWINTS.NUM_PROPOSALS 300
        
        self.hidden_dim = cfg.MODEL.SWINTS.HIDDEN_DIM
        # cfg.MODEL.SWINTS.HIDDEN_DIM 256
        
        self.num_heads = cfg.MODEL.SWINTS.NUM_HEADS
        #  cfg.MODEL.SWINTS.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        # self.backbone 스윈 트랜스포머+FPN
        
        self.size_divisibility = self.backbone.size_divisibility
        # self.size_divisibility 32
        
        # Build Proposals.
        self.pos_embeddings = nn.Embedding(self.num_proposals, self.hidden_dim)
        # self.pos_embeddings Embedding(300, 256)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        # self.init_proposal_boxes.shape = Embedding(300, 4)
        
        
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)



        # --------
        ## 이미지 특징 추출기
        self.IFE = ImgFeatExtractor(cfg)
        self.mask_encoding = PCAMaskEncoding(cfg)
        
        # encoding parameters.
        components_path = cfg.MODEL.SWINTS.PATH_COMPONENTS
        # update parameters.
        parameters = np.load(components_path)
        components = nn.Parameter(torch.from_numpy(parameters['components_c'][0]).float().to(self.device),requires_grad=False)
        explained_variances = nn.Parameter(torch.from_numpy(parameters['explained_variance_c'][0]).float().to(self.device), requires_grad=False)
        means = nn.Parameter(torch.from_numpy(parameters['mean_c'][0]).float().to(self.device),requires_grad=False)
        self.mask_encoding.components = components
        self.mask_encoding.explained_variances = explained_variances
        self.mask_encoding.means = means
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.SWINTS.CLASS_WEIGHT
        # cfg.MODEL.SWINTS.CLASS_WEIGHT 2.0
        
        giou_weight = cfg.MODEL.SWINTS.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SWINTS.L1_WEIGHT
        rec_weight = cfg.MODEL.SWINTS.REC_WEIGHT
        no_object_weight = cfg.MODEL.SWINTS.NO_OBJECT_WEIGHT
        mask_weight = cfg.MODEL.SWINTS.MASK_WEIGHT

        self.deep_supervision = cfg.MODEL.SWINTS.DEEP_SUPERVISION

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   cost_mask=mask_weight)
        self.matcher = matcher
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight, "loss_feat": mask_weight, "loss_dice": mask_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        weight_dict["loss_rec"] = rec_weight
        losses = ["labels", "boxes", "masks", "rec"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """        
        ## 픽셀 노말라이즈된 이미지, [이미지 너비,높이,너비,높이]
        images, images_whwh = self.preprocess_image(batched_inputs)
        
        # from torchvision import transforms
        # pillow_img = transforms.ToPILImage()
        # pillow_img(images.tensor[0]).save("./temp.png")
        
        ## 이미지를 텐서로 
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        # images.tensor.shape
        # torch.Size([1, 3, 768, 960])

    
        ## 백본네트워크를 통해 특징 추출
        # Feature Extraction.
        src = self.backbone(images.tensor)
        # src['p2'].shape
        # torch.Size([1, 256, 192, 240])
        # src['p3'].shape
        # torch.Size([1, 256, 96, 120])
        # src['p4'].shape
        # torch.Size([1, 256, 48, 60])
        # src['p5'].shape
        # torch.Size([1, 256, 24, 30])
        # src['p6'].shape
        # torch.Size([1, 256, 12, 15])
        # 피라미드 p1은 안쓴 이유는? 계산값이 비싸서
        


        # Prepare Proposals.센터좌표 x,y, 박스 가로세로 비율
        proposal_boxes = self.init_proposal_boxes.weight.clone()
    #     tensor([[0.5111, 0.5021, 0.9991, 0.9842],
    #     [0.4948, 0.4812, 1.0185, 0.9961],
    #     [0.4904, 0.4460, 1.0025, 1.0052],
    #     ...,
    #     [0.4827, 0.4773, 0.9672, 0.9317],
    #     [0.5092, 0.4971, 0.9541, 0.9551],
    #     [0.5253, 0.5356, 0.9979, 1.0019]], device='cuda:0',
    #    grad_fn=<CloneBackward>)
        # proposal_boxes torch.Size([300, 4])
        
        ## 센터좌표,너비,높이를 사각형 시작점(x1,y1),끝점(x2,y2)로 바꾸어줌
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
        


        ## p6는 안씀
        features = list()      
        for f in self.in_features:
            feature = src[f]
            features.append(feature)
        
        
        ## 스윈트랜스포머에서 뽑아온 피쳐(features)를 p2,p3,p4,p5를 평균값 취함
        ## proposal_feats임 -> 포지션 임베딩까지 더해줘야함
        ## 초기화는 이미지 피쳐를 평균값 + 더함 -> 복사 
        img_feats = self.IFE(features)
        #img_feats =  torch.Size([1, 300, 256])
        
        
        ## 배치사이즈
        bs = len(features[0])
        
        ## 포지션 임베딩 웨이트 복사
        pos_embeddings = self.pos_embeddings.weight[None].repeat(bs, 1, 1)
        
        ## 특징값 + 포지션 임베딩값 
        proposal_feats = img_feats + pos_embeddings
        
        
        
        del img_feats
        
        # self.training = False
        if self.training:
            ## instance 자료 구조
            ## 이미지 너비,높이, GT box 좌표,box/4나눈 좌표, GT 클래스(문자영역=0),마스크맵,문자            
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            ## 데이터셋에서 진짜 필요한 정보만 파싱 (기본이 디텍트론데이터셋형식을 따름)
            targets = self.prepare_targets(gt_instances)
            """ 
            타겟 정보
            dict_keys(['labels', 'boxes', 'boxes_xyxy', 'image_size_xyxy', 'image_size_xyxy_tgt', 'area', 'gt_masks', 'rec'])
            
            'labels': tensor([0, 0, 0, 0, 0, 0, 0], device='cuda:0')
            
            'boxes':
            tensor([[0.3373, 0.3555, 0.0912, 0.1085],
                    [0.4065, 0.4614, 0.1771, 0.1758],
                    [0.5241, 0.5475, 0.1107, 0.1082],
                    [0.6197, 0.5558, 0.1093, 0.1273],
                    [0.7042, 0.5884, 0.0479, 0.0507],
                    [0.7512, 0.6045, 0.0690, 0.0683],
                    [0.7084, 0.6138, 0.0653, 0.0530]], device='cuda:0')
                    
                    
            'boxes_xyxy':
            tensor([[277.7254, 231.3336, 364.5794, 314.6411],
                    [302.7027, 286.8394, 471.3041, 421.8863],
                    [446.3011, 378.8896, 551.6727, 462.0249],
                    [537.9084, 378.0179, 641.9225, 475.7526],
                    [647.6205, 432.4008, 693.2263, 471.3675],
                    [682.2961, 438.0428, 748.0281, 490.5096],
                    [643.2863, 451.0175, 705.4629, 491.7579]], device='cuda:0')
                    
                    
            'image_size_xyxy':
            tensor([952., 768., 952., 768.], device='cuda:0')
            
            
            'image_size_xyxy_tgt':
            tensor([[952., 768., 952., 768.],
                    [952., 768., 952., 768.],
                    [952., 768., 952., 768.],
                    [952., 768., 952., 768.],
                    [952., 768., 952., 768.],
                    [952., 768., 952., 768.],
                    [952., 768., 952., 768.]], device='cuda:0')
                    
                    
            'area':
            tensor([ 7235.5894, 22769.1035,  8760.0947, 10165.7939,  1777.1068,  3448.7432,
                    2533.0977], device='cuda:0')
                    
                    
            'gt_masks':
            tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
            
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                ...
                
            'rec':
            tensor([[80, 77, 69,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,

            """
            
            
            outputs_class, outputs_coord, outputs_mask,out_rec = self.head(features, proposal_boxes, proposal_feats, targets, mask_encoding=self.mask_encoding, matcher=self.matcher)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_masks': outputs_mask[-1], 'pred_rec': out_rec}
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                                         for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])]

            loss_dict = self.criterion(output, targets, self.mask_encoding)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            outputs_class, outputs_coord, outputs_mask,out_rec = self.head(features, proposal_boxes, proposal_feats, mask_encoding=self.mask_encoding)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_masks': outputs_mask[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"].unsqueeze(dim=2)
            results = Instances(images.image_sizes[0])
            results.pred_boxes = Boxes(box_pred)
            results.scores = box_cls # 박스 클래스가 문자열 디텍션 점수같은 개념
            results.pred_masks = mask_pred.squeeze(1)
            results.pred_rec = out_rec
            results = [results]
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            # import cv2
            # img = cv2.imread(batched_inputs[0]['file_name'])
            
            
            return processed_results

    # 데이터셋에서 필요한 정보만 가져오기
    @torch.no_grad()
    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)

            target["gt_masks"] = targets_per_image.gt_masks.to(self.device)
            masks = target['gt_masks'].crop_and_resize(targets_per_image.gt_boxes, 28)
            target["gt_masks"] = masks.float()
            target["rec"] = targets_per_image.rec.to(self.device)
            new_targets.append(target)

        return new_targets

    @torch.no_grad()
    def inference(self, box_cls, box_pred, mask_pred, image_sizes, recred):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        #
        scores = torch.sigmoid(box_cls)
        labels = torch.arange(self.num_classes, device=self.device).\
                 unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
        for i, (scores_per_image, box_pred_per_image, mask_pred_per_image, image_size, rec_per_image) in enumerate(zip(
                scores, box_pred, mask_pred, image_sizes, rec_pred
        )):
            result = Instances(image_size)
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
            labels_per_image = labels[topk_indices]
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.pred_masks = mask_pred_per_image
            result.pred_rec = rec_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
