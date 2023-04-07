# Standard Library
import json
import copy
from collections import Counter, OrderedDict

# Import from third library
import torch
import numpy as np
import pandas
import yaml
from prettytable import PrettyTable

from up.utils.general.log_helper import default_logger as logger
from up.utils.general.yaml_loader import IncludeLoader
from up.utils.general.registry_factory import EVALUATOR_REGISTRY
from up.utils.general.petrel_helper import PetrelHelper
from up.data.metrics.base_evaluator import Evaluator, Metric
from up.tasks.det.data.metrics.custom_evaluator import CustomEvaluator

__all__ = ["DacEvaluator"]


def box_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

@EVALUATOR_REGISTRY.register('dac')
class DacEvaluator(CustomEvaluator):
    """Calculate mAP&MR@FPPI for custom dataset"""

    def __init__(self,
                 gt_file,
                 num_classes,
                 iou_thresh,
                 class_names=None,
                 fppi=np.array([0.1, 0.5, 1]),
                 metrics_csv='metrics.csv',
                 label_mapping=None,
                 ignore_mode=0,
                 ign_iou_thresh=0.5,
                 iou_types=['bbox'],
                 eval_class_idxs=[],
                 cross_cfg=None):

        super(DacEvaluator, self).__init__(gt_file,
                                          num_classes,
                                          iou_thresh,
                                          metrics_csv=metrics_csv,
                                          label_mapping=label_mapping,
                                          ignore_mode=ignore_mode,
                                          ign_iou_thresh=ign_iou_thresh,
                                          cross_cfg=cross_cfg)

        if len(eval_class_idxs) == 0:
            eval_class_idxs = list(range(1, num_classes))
        self.eval_class_idxs = eval_class_idxs

        self.fppi = np.array(fppi)
        self.class_names = class_names
        self.metrics_csv = metrics_csv
        if self.class_names is None:
            self.class_names = eval_class_idxs
        self.iou_types = iou_types


    def load_gts(self, gt_files):
        # maintain a dict to store original img information
        # key is image dir,value is image_height,image_width,instances
        original_gt = {}
        gts = {
            'bbox_num': 0, # Counter(),
            'gt_num': 0, # Counter(),
            'image_num': 0,
            'image_ids': list()
        }
        if not isinstance(gt_files, list):
            gt_files = [gt_files]
        for gt_file_idx, gt_file in enumerate(gt_files):
            gt_img_ids = set()
            with PetrelHelper.open(gt_file) as f:
                for i, line in enumerate(f):
                    img = json.loads(line)
                    # if self.label_mapping is not None:
                    #     img = self.set_label_mapping(img, gt_file_idx)
                    # if self.cross_cfg is not None:
                    #     img = self.set_label_mapping(img, gt_file_idx)
                    image_id = img['filename']
                    original_gt[img['filename']] = copy.deepcopy(img)
                    gt_img_ids.add(image_id)
                    gts['image_num'] += 1
                    for idx, instance in enumerate(img.get('instances', [])):
                        instance['detected'] = False
                        # remember the original index within an image of annoated format so
                        # we can recover from distributed format into original format
                        is_ignore = instance.get('is_ignored', False)
                        instance['local_index'] = idx
                        # label = instance.get('label', 0)
                        # ingore mode
                        # 0 indicates all classes share ignore region, label is set to -1
                        # 1 indicates different classes different ignore region, ignore label must be provided
                        # 2 indicates we ingore all ignore regions
                        # if is_ignore and self.ignore_mode == 0:
                        #     label = -1
                        # box_by_label = gts.setdefault(label, {})
                        # box_by_img = box_by_label.setdefault(image_id, {'gts': []})
                        box_by_img = gts.setdefault(image_id, {'gts': []})
                        gt_by_img = box_by_img['gts']
                        # gts['bbox_num'][label] += 1
                        gts['bbox_num'] += 1
                        if not is_ignore:
                            gt_by_img.append(instance)
                            # gts['gt_num'][label] += 1
                            gts['gt_num'] += 1
                        else:
                            ign_by_img = box_by_img.setdefault('ignores', [])
                            ign_by_img.append(instance)
                gts['image_ids'].append(gt_img_ids)
        return gts, original_gt

    def load_dts(self, res_file, res=None):
        dts = {}
        if res is None:
            logger.info(f'loading res from {res_file}')
            with open(res_file, 'r') as f:
                for line in f:
                    dt = json.loads(line)
                    dt_by_img = dts.setdefault(dt['image_id'], [])
                    dt_by_img.append(dt)
                    # dt_by_label = dts.setdefault(dt['label'], [])
                    # dt_by_label.append(dt)
        else:
            for device_res in res:
                for lines in device_res:
                    for line in lines:
                        dt_by_img = dts.setdefault(line['image_id'], [])
                        dt_by_img.append(line)
                        # dt_by_label = dts.setdefault(line['label'], [])
                        # dt_by_label.append(line)
        return dts

    def eval(self, res_file, res=None):
        # from up.data.metrics.base_evaluator import Evaluator, Metric
        metric_res = Metric({})
        self.gts, original_gt = self.load_gts(self.gt_file)
        dts = self.load_dts(res_file, res)
        image_ids = list(self.gts['image_ids'][0])
        ious = torch.zeros(len(image_ids))
        for idx in range(len(image_ids)):
            img_id = image_ids[idx]
            gt_bboxes_item = self.gts[img_id]['gts'][0]
            dt_bboxes_item = dts[img_id][0]
            gt_bboxes = gt_bboxes_item['bbox']
            dt_bboxes = dt_bboxes_item['bbox']
            image_info = dt_bboxes_item['image_info']
            ratio_h, ratio_w = image_info[2]
            img_h_r, img_w_r = image_info[0], image_info[1]
            image_height, image_width = image_info[3], image_info[4] 
            feat_h, feat_w = dt_bboxes_item['feat_H'], dt_bboxes_item['feat_W']
            stride = img_h_r / feat_h
            gt_bboxes_scale = copy.copy(gt_bboxes)
            gt_bboxes_scale[0] = (gt_bboxes[0] + gt_bboxes[2]) / 2
            gt_bboxes_scale[1] = (gt_bboxes[1] + gt_bboxes[3]) / 2
            gt_bboxes_scale[2] = gt_bboxes[2] - gt_bboxes[0]
            gt_bboxes_scale[3] = gt_bboxes[3] - gt_bboxes[1]
            dt_bboxes[0] = dt_bboxes[0] / ratio_w * stride
            dt_bboxes[1] = dt_bboxes[1] / ratio_h * stride
            dt_bboxes[2] = dt_bboxes[2] / ratio_w * stride
            dt_bboxes[3] = dt_bboxes[3] / ratio_h * stride
            dt_bboxes_xywh = copy.copy(dt_bboxes)
            dt_bboxes_xywh[0] = (dt_bboxes[0] + dt_bboxes[2]) / 2
            dt_bboxes_xywh[1] = (dt_bboxes[1] + dt_bboxes[3]) / 2
            dt_bboxes_xywh[2] = dt_bboxes[2] - dt_bboxes[0]
            dt_bboxes_xywh[3] = dt_bboxes[3] - dt_bboxes[1]
            dt_bboxes_xywh = torch.tensor(dt_bboxes_xywh).cuda()
            gt_bboxes_scale = torch.tensor(gt_bboxes_scale).cuda()
            ious[idx] = box_iou(dt_bboxes_xywh, gt_bboxes_scale, x1y1x2y2=False).item()
        mean_ious = torch.mean(ious).item()
        metric_res.update({'mean_ious': mean_ious})

        return metric_res

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for CustomDataset of mAP metric')
        subparser.add_argument('--gt_file', required=True, help='annotation file')
        subparser.add_argument('--res_file', required=True, help='results file of detection')
        subparser.add_argument('--num_classes',
                               type=int,
                               default=None,
                               help='number of classes including __background__ class')
        subparser.add_argument('--class_names',
                               type=lambda x: x.split(','),
                               default=None,
                               help='names of classes including __background__ class')
        subparser.add_argument('--iou_thresh',
                               type=float,
                               default=0.5,
                               help='iou thresh to classify true positives & false postitives')
        subparser.add_argument('--ign_iou_thresh',
                               type=float,
                               default=0.5,
                               help='ig iou thresh to ignore false postitives')
        subparser.add_argument('--metrics_csv',
                               type=str,
                               default='metrics.csv',
                               help='file to save evaluation metrics')
        subparser.add_argument('--img_root',
                               type=str,
                               default='None',
                               help='directory of images used to evaluate, only used for visualization')
        subparser.add_argument('--bad_case_analyser',
                               type=str,
                               default='manual_0.5',
                               help='choice of criterion for analysing bad case, format:{manual or fppi}_{[score]}')
        subparser.add_argument('--vis_mode',
                               type=str,
                               default=None,
                               choices=['all', 'fp', 'fn', None],
                               help='visualize fase negatives or fase positive or all')
        subparser.add_argument('--eval_class_idxs',
                               type=lambda x: list(map(int, x.split(','))),
                               default=[],
                               help='eval subset of all classes, 1,3,4'
                               )
        subparser.add_argument('--ignore_mode',
                               type=int,
                               default=0,
                               help='ignore mode, default as 0')
        subparser.add_argument('--config', type=str, default='', help='training config for eval')
        subparser.add_argument('--iou_types',
                               type=list,
                               default=['bbox'],
                               help='iou type to select eval mode')

        return subparser

    @classmethod
    def get_classes(self, gt_file):
        all_classes = set()

        with PetrelHelper.open(gt_file) as f:
            for line in f:
                data = json.loads(line)
                labels = set([ins['label'] for ins in data['instances'] if ins['label'] > 0])
                all_classes |= labels
        all_classes = [0] + sorted(list(all_classes))
        class_names = [str(_) for _ in all_classes]
        print('class_names:{}'.format(class_names))
        return class_names

    @classmethod
    def from_args(cls, args):
        if args.config != '':   # load from training config
            cfg = yaml.load(open(args.config, 'r'), Loader=IncludeLoader)
            eval_kwargs = cfg['dataset']['test']['dataset']['kwargs']['evaluator']['kwargs']
            eval_kwargs['metrics_csv'] = args.metrics_csv
            return cls(**eval_kwargs)
        if args.num_classes is None:
            args.class_names = cls.get_classes(args.gt_file)
            args.num_classes = len(args.class_names)
        return cls(args.gt_file,
                   args.num_classes,
                   args.class_names,
                   args.iou_thresh,
                   metrics_csv=args.metrics_csv,
                   ignore_mode=args.ignore_mode,
                   eval_class_idxs=args.eval_class_idxs,
                   ign_iou_thresh=args.ign_iou_thresh,
                   iou_types=args.iou_types,)





@EVALUATOR_REGISTRY.register('dac_retina')
class DacRetinaEvaluator(Evaluator):
    def __init__(self, gt_file, num_classes, iou_thresh=0.5, ign_iou_thresh=0.5, metrics_csv='metrics.csv', label_mapping=None, ignore_mode=0, cross_cfg=None):  # noqa
        super(DacRetinaEvaluator, self).__init__()
        self.gt_file = gt_file
        self.iou_thresh = iou_thresh
        self.ign_iou_thresh = ign_iou_thresh
        self.num_classes = num_classes
        self.gt_loaded = False
        self.metrics_csv = metrics_csv
        self.label_mapping = label_mapping
        self.class_from = {}
        self.cross_cfg = cross_cfg
        if self.cross_cfg is not None:
            self.label_mapping = self.cross_cfg.get('label_mapping', [[]])
        if self.label_mapping is not None:
            for idx, label_map in enumerate(self.label_mapping):
                for label in label_map:
                    self.class_from[label] = [idx]
        else:
            self.label_mapping = [[]]
            if not isinstance(gt_file, list):
                gt_file = [gt_file]
            for label in range(1, self.num_classes):
                self.class_from[label] = list(range(len(gt_file)))
        self.ignore_mode = ignore_mode

    def set_label_mapping(self, data, idx):
        if len(self.label_mapping[0]) == 0:
            return data
        instances = data.get('instances', [])
        for instance in instances:
            if instance.get("is_ignored", False):
                if self.ignore_mode == 1:
                    pass
                else:
                    continue
            instance["label"] = self.label_mapping[idx][int(instance["label"] - 1)]
        return data

    def load_gts(self, gt_files):
        # maintain a dict to store original img information
        # key is image dir,value is image_height,image_width,instances
        original_gt = {}
        gts = {
            'bbox_num': 0, # Counter(),
            'gt_num': 0, # Counter(),
            'image_num': 0,
            'image_ids': list()
        }
        if not isinstance(gt_files, list):
            gt_files = [gt_files]
        for gt_file_idx, gt_file in enumerate(gt_files):
            gt_img_ids = set()
            with PetrelHelper.open(gt_file) as f:
                for i, line in enumerate(f):
                    img = json.loads(line)
                    # if self.label_mapping is not None:
                    #     img = self.set_label_mapping(img, gt_file_idx)
                    # if self.cross_cfg is not None:
                    #     img = self.set_label_mapping(img, gt_file_idx)
                    image_id = img['filename']
                    original_gt[img['filename']] = copy.deepcopy(img)
                    gt_img_ids.add(image_id)
                    gts['image_num'] += 1
                    for idx, instance in enumerate(img.get('instances', [])):
                        instance['detected'] = False
                        # remember the original index within an image of annoated format so
                        # we can recover from distributed format into original format
                        is_ignore = instance.get('is_ignored', False)
                        instance['local_index'] = idx
                        # label = instance.get('label', 0)
                        # ingore mode
                        # 0 indicates all classes share ignore region, label is set to -1
                        # 1 indicates different classes different ignore region, ignore label must be provided
                        # 2 indicates we ingore all ignore regions
                        # if is_ignore and self.ignore_mode == 0:
                        #     label = -1
                        # box_by_label = gts.setdefault(label, {})
                        # box_by_img = box_by_label.setdefault(image_id, {'gts': []})
                        box_by_img = gts.setdefault(image_id, {'gts': []})
                        gt_by_img = box_by_img['gts']
                        # gts['bbox_num'][label] += 1
                        gts['bbox_num'] += 1
                        if not is_ignore:
                            gt_by_img.append(instance)
                            # gts['gt_num'][label] += 1
                            gts['gt_num'] += 1
                        else:
                            ign_by_img = box_by_img.setdefault('ignores', [])
                            ign_by_img.append(instance)
                gts['image_ids'].append(gt_img_ids)
        return gts, original_gt

    def load_dts(self, res_file, res=None):
        dts = {}
        if res is None:
            logger.info(f'loading res from {res_file}')
            with open(res_file, 'r') as f:
                for line in f:
                    dt = json.loads(line)
                    dt_by_img = dts.setdefault(dt['image_id'], [])
                    dt_by_img.append(dt)
                    # dt_by_label = dts.setdefault(dt['label'], [])
                    # dt_by_label.append(dt)
        else:
            for device_res in res:
                for lines in device_res:
                    for line in lines:
                        dt_by_img = dts.setdefault(line['image_id'], [])
                        dt_by_img.append(line)
                        # dt_by_label = dts.setdefault(line['label'], [])
                        # dt_by_label.append(line)
        return dts

    def eval(self, res_file, res=None):
        # from up.data.metrics.base_evaluator import Evaluator, Metric
        metric_res = Metric({})
        self.gts, original_gt = self.load_gts(self.gt_file)
        dts = self.load_dts(res_file, res)
        image_ids = list(self.gts['image_ids'][0])
        ious = torch.zeros(len(image_ids))
        for idx in range(len(image_ids)):
            img_id = image_ids[idx]
            gt_bboxes_item = self.gts[img_id]['gts'][0]
            dt_bboxes_item = dts[img_id][0]
            gt_bboxes = gt_bboxes_item['bbox']
            dt_bboxes = dt_bboxes_item['bbox']
            # image_info = dt_bboxes_item['image_info']
            # ratio_h, ratio_w = image_info[2]

            gt_bboxes[0] = gt_bboxes[0] # / ratio_w
            gt_bboxes[1] = gt_bboxes[1] # / ratio_h
            gt_bboxes[2] = gt_bboxes[2] # / ratio_w
            gt_bboxes[3] = gt_bboxes[3] # / ratio_h
            # image_height, image_width = image_info[3], image_info[4] 
            # feat_h, feat_w = dt_bboxes_item['feat_H'], dt_bboxes_item['feat_W']
            # gt_bboxes_scale = copy.copy(gt_bboxes)
            # gt_bboxes_scale[0] = gt_bboxes[0] * feat_w / image_width
            # gt_bboxes_scale[1] = gt_bboxes[1] * feat_h / image_height
            # gt_bboxes_scale[2] = gt_bboxes[2] * feat_w / image_width
            # gt_bboxes_scale[3] = gt_bboxes[3] * feat_h / image_height
            dt_bboxes[0] = dt_bboxes[0] # / ratio_w
            dt_bboxes[1] = dt_bboxes[1] # / ratio_h
            dt_bboxes[2] = dt_bboxes[2] # / ratio_w
            dt_bboxes[3] = dt_bboxes[3] # / ratio_h
            # dt_bboxes_xywh = copy.copy(dt_bboxes)
            # dt_bboxes_xywh[0] = (dt_bboxes[0] + dt_bboxes[2]) / 2
            # dt_bboxes_xywh[1] = (dt_bboxes[1] + dt_bboxes[3]) / 2
            # dt_bboxes_xywh[2] = dt_bboxes[2] - dt_bboxes[0]
            # dt_bboxes_xywh[3] = dt_bboxes[3] - dt_bboxes[1]
            dt_bboxes = torch.tensor(dt_bboxes).cuda()
            gt_bboxes = torch.tensor(gt_bboxes).cuda()
            ious[idx] = box_iou(dt_bboxes, gt_bboxes, x1y1x2y2=True).item()
        mean_ious = torch.mean(ious).item()
        metric_res.update({'mean_ious': mean_ious})
        metric_res.set_cmp_key('mean_ious')

        return metric_res

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for CustomDataset of mAP metric')
        subparser.add_argument('--gt_file', required=True, help='annotation file')
        subparser.add_argument('--res_file', required=True, help='results file of detection')
        subparser.add_argument('--num_classes',
                               type=int,
                               default=None,
                               help='number of classes including __background__ class')
        subparser.add_argument('--class_names',
                               type=lambda x: x.split(','),
                               default=None,
                               help='names of classes including __background__ class')
        subparser.add_argument('--iou_thresh',
                               type=float,
                               default=0.5,
                               help='iou thresh to classify true positives & false postitives')
        subparser.add_argument('--ign_iou_thresh',
                               type=float,
                               default=0.5,
                               help='ig iou thresh to ignore false postitives')
        subparser.add_argument('--metrics_csv',
                               type=str,
                               default='metrics.csv',
                               help='file to save evaluation metrics')
        subparser.add_argument('--img_root',
                               type=str,
                               default='None',
                               help='directory of images used to evaluate, only used for visualization')
        subparser.add_argument('--bad_case_analyser',
                               type=str,
                               default='manual_0.5',
                               help='choice of criterion for analysing bad case, format:{manual or fppi}_{[score]}')
        subparser.add_argument('--vis_mode',
                               type=str,
                               default=None,
                               choices=['all', 'fp', 'fn', None],
                               help='visualize fase negatives or fase positive or all')
        subparser.add_argument('--eval_class_idxs',
                               type=lambda x: list(map(int, x.split(','))),
                               default=[],
                               help='eval subset of all classes, 1,3,4'
                               )
        subparser.add_argument('--ignore_mode',
                               type=int,
                               default=0,
                               help='ignore mode, default as 0')
        subparser.add_argument('--config', type=str, default='', help='training config for eval')
        subparser.add_argument('--iou_types',
                               type=list,
                               default=['bbox'],
                               help='iou type to select eval mode')

        return subparser

    @classmethod
    def get_classes(self, gt_file):
        all_classes = set()

        with PetrelHelper.open(gt_file) as f:
            for line in f:
                data = json.loads(line)
                labels = set([ins['label'] for ins in data['instances'] if ins['label'] > 0])
                all_classes |= labels
        all_classes = [0] + sorted(list(all_classes))
        class_names = [str(_) for _ in all_classes]
        print('class_names:{}'.format(class_names))
        return class_names

    @classmethod
    def from_args(cls, args):
        if args.config != '':   # load from training config
            cfg = yaml.load(open(args.config, 'r'), Loader=IncludeLoader)
            eval_kwargs = cfg['dataset']['test']['dataset']['kwargs']['evaluator']['kwargs']
            eval_kwargs['metrics_csv'] = args.metrics_csv
            return cls(**eval_kwargs)
        if args.num_classes is None:
            args.class_names = cls.get_classes(args.gt_file)
            args.num_classes = len(args.class_names)
        return cls(args.gt_file,
                   args.num_classes,
                   args.class_names,
                   args.iou_thresh,
                   metrics_csv=args.metrics_csv,
                   ignore_mode=args.ignore_mode,
                   eval_class_idxs=args.eval_class_idxs,
                   ign_iou_thresh=args.ign_iou_thresh,
                   iou_types=args.iou_types,)
