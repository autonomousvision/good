"""This file contains code to build dataloader of COCO-split dataset.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import imp
import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
# from pycocotools.coco import COCO
# from .coco_api import COCO
from .api_wrappers import COCO, COCOeval
# Added for cross-category evaluation
from .cocoeval_wrappers import COCOEvalWrapper, COCOEvalXclassWrapper

from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .coco import CocoDataset

import warnings

try:
    import pycocotools
    # if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
    #     assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')


@DATASETS.register_module()
class CocoSplitDataset(CocoDataset):

    def __init__(self, 
                 is_class_agnostic=False, 
                 train_class='all',
                 eval_class='all',
                 **kwargs):
        # We convert all category IDs into 1 for the class-agnostic training and
        # evaluation. We train on train_class and evaluate on eval_class split.
        self.is_class_agnostic = is_class_agnostic
        self.train_class = train_class
        self.eval_class = eval_class
        super(CocoSplitDataset, self).__init__(**kwargs)
    
        if self.is_class_agnostic:
            self.CLASSES = ('foreground', )

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    VOC_CLASSES = (
               'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 
               'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
               'train', 'tv')
    NONVOC_CLASSES = (
               'truck', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench',
               'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake',
               'bed', 'toilet', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    class_names_dict = {
        'all': CLASSES,
        'voc': VOC_CLASSES,
        'nonvoc': NONVOC_CLASSES
    }

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)

        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.train_cat_ids = self.coco.getCatIds(
            catNms=self.class_names_dict[self.train_class]
            )
        self.eval_cat_ids = self.coco.getCatIds(
            catNms=self.class_names_dict[self.eval_class]
            )
        if self.is_class_agnostic: # Training time option; already not used in my experiments
            warnings.warn("You are using class agnostic training option. It's used in OLN. But already deprecated for open-set experiments.")
            self.cat2label = {cat_id: 0 for cat_id in self.cat_ids}
        else:
            self.cat2label = {
                cat_id: i for i, cat_id in enumerate(self.cat_ids)}                    

        self.img_ids = self.coco.getImgIds()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    # Refer to custom.py -- filter_img is not used in test_mode.
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        
        for i, class_id in enumerate(self.train_cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.
        Used through "get_ann_info" only during training.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        TODO: The eval_bboxes and etc are no longer used. Relevant code can be deleted.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        eval_bboxes = []
        eval_labels = []
        eval_bboxes_ignore = []
        eval_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            if (ann['category_id'] not in self.eval_cat_ids) and (ann['category_id'] not in self.train_cat_ids):
                continue

            if ann['category_id'] in self.train_cat_ids:
                bbox = [x1, y1, x1 + w, y1 + h]
                if ann.get('iscrowd', False):
                    gt_bboxes_ignore.append(bbox)
                else:
                    gt_bboxes.append(bbox)                
                    gt_labels.append(self.cat2label[ann['category_id']])
                    gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        try:
            seg_map = img_info['filename'].replace('jpg', 'png')
        except:
            seg_map = img_info['filename']

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                #  proposal_nums=(10, 20, 30, 50, 100, 300, 500, 1000, 1500),
                proposal_nums=(10, 20, 30, 50, 100, 300, 500),
                # proposal_nums=(10, 50, 100, 300),
                 iou_thrs=None,
                 metric_items=None,
                 low_max_det=False):
        """Evaluation in COCO-Split protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        if classwise or low_max_det:
            proposal_nums=(1, 3, 5, 10, 100, 300, 500)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]
        print('formatting results')
        test_result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        print('results formatted')

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in test_result_files:
                raise KeyError(f'{metric} is not in test results')
            try:
                cocoDt = cocoGt.loadRes(test_result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric

            # Class manipulation.
            for idx, ann in enumerate(cocoGt.dataset['annotations']):
                if ann['category_id'] in self.eval_cat_ids:
                    cocoGt.dataset['annotations'][idx]['ignored_split'] = 0
                else:
                    cocoGt.dataset['annotations'][idx]['ignored_split'] = 1

            # Cross-category evaluation wrapper.
            cocoEval = COCOEvalXclassWrapper(cocoGt, cocoDt, iou_type)

            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.traincatIds = self.train_cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            cocoEval.params.classwise = classwise
            # mapping of cocoEval.stats
            if classwise or low_max_det:
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@1': 6,
                    'AR@3': 7,
                    'AR@5':8,
                    'AR@10': 9,
                    'AR@100': 10,
                    'AR@300': 11,
                    'AR@10_s':12,
                    'AR@10_m':13,
                    'AR@10_l':14,
                    # 'AR@500': 12,
                    # 'AR@1000': 13,
                    # 'AR@1500': 14,
                }
            else:
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@10': 6,
                    'AR@20': 7,
                    'AR@30':8,
                    'AR@50': 9,
                    'AR@100': 10,
                    'AR@300': 11,
                    'AR@100_s':13,
                    'AR@100_m':14,
                    'AR@100_l':15,
                }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')
            cocoEval.params.useCats = 0  # treat all FG classes as single class.
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP XXX: Can tell us AP of each category, but not necessary for calculating mAP; mAP calculation happens in cocoEval
                # In class agnostic evaluation, per-category AP is not recorded
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                recalls = cocoEval.eval['recall'] # T,K,A,M
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                AR100_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    # recall = recalls[:, idx, 0, 4] # AR100
                    # recall = recalls[:, idx, 0, 3] # AR10
                    recall = recalls[:, idx, 0, 2] # AR5
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    if recall.size:
                        rec = np.mean(recall)
                    else:
                        rec = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))
                    AR100_per_category.append(
                        (f'{nm["name"]}', f'{float(rec):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

                num_columns = min(6, len(AR100_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*AR100_per_category))
                headers = ['category', 'AR100'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)
                print('\n AR100_per_category:', AR100_per_category)

            if metric_items is None:
                if classwise or low_max_det:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 
                        'AR@1', 'AR@3', 'AR@5', 'AR@10', 'AR@100', 'AR@300',
                        # 'AR@500', 'AR@1000', 'AR@1500', 
                        'AR@10_s', 'AR@10_m', 'AR@10_l',
                    ]
                else:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 
                        'AR@10', 'AR@20', 'AR@50', 'AR@100', 'AR@300',
                        'AR@100_s', 'AR@100_m', 'AR@100_l',
                    ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
            
            if classwise or low_max_det:
                print_metrics = ['AR@1', 'AR@3', 'AR@5', 'AR@10', 'AR@100', 'AR@10_s', 'AR@10_m', 'AR@10_l']
            else:
                print_metrics = ['AR@10', 'AR@100', 'mAP', 'AR@100_s', 'AR@100_m', 'AR@100_l']
            print_str = ""
            for metric_item in print_metrics:
                val = f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                print_str += val + ","
            print(print_str)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
    
    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))
        # if len(results) != len(self):
        #     warnings.warn(f'The length of results is not equal to the dataset len: {len(results)} != {len(self)}')

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir
    
    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def _det2json(self, results):
        """Convert detection results to COCO json style. But with key "logit"."""
        json_results = []
        for idx in range(len(self)):
            try:
                img_id = self.img_ids[idx]
                result = results[idx]
                for label in range(len(result)):
                    bboxes = result[label]
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data['image_id'] = img_id
                        data['bbox'] = self.xyxy2xywh(bboxes[i])
                        data['score'] = float(bboxes[i][4])
                        data['ood_score'] = float(bboxes[i][-1]) # float(bboxes[i][5])
                        data['category_id'] = self.cat_ids[label] # label number will always be smaller than len(self.train_cat_ids)
                        json_results.append(data)
            except:
                # print(f'cant load img/result id {idx}')
                continue
        return json_results
    
    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['logit'] = float(bboxes[i][-1])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['logit'] = float(mask_score[i]) # XXX: Temporary. Not real logits.
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results
    
    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['logit'] = float(bboxes[i][-1])
                data['category_id'] = 1
                json_results.append(data)
        return json_results
