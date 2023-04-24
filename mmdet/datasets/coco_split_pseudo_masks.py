"""This file contains code to build dataloader of COCO-split pseudo box dataset.
"""
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import random

from .api_wrappers import COCO

from .builder import DATASETS
from .coco_split import CocoSplitDataset
import torch 
from mmcv.ops.nms import batched_nms

@DATASETS.register_module()
class CocoSplitPseudoBoxDataset(CocoSplitDataset):
    """
    Used to joint train on images with both pseudo-GT and GT.
    """

    def __init__(
        self,
        additional_ann_file=None,
        iou_thresh=None,
        score_thresh=None,
        top_k=None,
        random_sample_masks=False,
        merge_nms=False,
        **kwargs,
    ):
        # Add additional annotation file (eg. from pseudo masks)
        self.additional_coco = None
        if additional_ann_file is not None:
            if isinstance(additional_ann_file, list):
                self.additional_coco =[COCO(add_ann_file) for add_ann_file in additional_ann_file]
            else:
                self.additional_coco = [COCO(additional_ann_file)]
            if isinstance(top_k, list):
                if len(top_k) == 1:
                    self.top_k = top_k * len(self.additional_coco)
                else:
                    self.top_k = top_k
                    assert len(top_k)==len(self.additional_coco), "Need to specify a topk for each additional ann file!"
            else:
                self.top_k = [top_k] * len(self.additional_coco)
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.random_sample_masks = random_sample_masks
        self.merge_nms = merge_nms
        super(CocoSplitPseudoBoxDataset, self).__init__(**kwargs)

    # Override to load pseudo masks
    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]["id"]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        all_anns = []
        all_anns.extend(ann_info)
        additional_anns = []
        if self.additional_coco is not None:
            for i, additional_coco in enumerate(self.additional_coco):
                additional_ann_ids = additional_coco.get_ann_ids(img_ids=[img_id])
                additional_ann_info = additional_coco.load_anns(additional_ann_ids)
                additional_ann_info = self.sample_topk(additional_ann_info, top_k=self.top_k[i])
                additional_anns.extend(additional_ann_info)
            # Filter pseudo boxes based on Iou overlapping
            if self.merge_nms:
                if len(additional_anns) > 1 and len(self.additional_coco)>1:
                    additional_anns = self.filter_overlaps(additional_anns)
            all_anns.extend(additional_anns)
        return self._parse_ann_info(self.data_infos[idx], all_anns)
    
    def filter_overlaps(self, annotations, iou_th=0.7):
        def _xywh2xyxy(bbox):
            x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
            return [x, y, x+w, y+h]
        nms_bboxes = torch.tensor([_xywh2xyxy(ann['bbox']) for ann in annotations])
        nms_scores = torch.tensor([ann['score'] for ann in annotations])
        nms_labels = torch.tensor([0] * len(annotations))
        nms_cfg = dict(type='nms', iou_threshold=0.7, class_agnostic=True)
        _, keep = batched_nms(nms_bboxes, nms_scores, nms_labels, nms_cfg)
        new_anns = [annotations[idx] for idx in keep]
        return new_anns

    def sample_topk(self, annotations, top_k=None):
        new_anns = annotations
        if self.iou_thresh is not None:
            tmp_new_anns = []
            for ann in new_anns:
                if ann["gt_iou"] < self.iou_thresh:
                    tmp_new_anns.append(ann)
            new_anns = tmp_new_anns
        if self.score_thresh is not None:
            tmp_new_anns = []
            for ann in new_anns:
                if ann["score"] >= self.score_thresh:
                    tmp_new_anns.append(ann)
            new_anns = tmp_new_anns
        if self.random_sample_masks:
            random.shuffle(new_anns)
        if top_k is not None:
            new_anns = new_anns[: top_k]
        return new_anns
