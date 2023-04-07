# Import from third library
import torch

from up.extensions import nms
# from up.tasks.det.models.utils.nms_wrapper import nms
from up.utils.general.fp16_helper import to_float32
from up.utils.general.registry_factory import ROI_MERGER_REGISTRY, ROI_PREDICTOR_REGISTRY
from up.tasks.det.models.utils.bbox_helper import (
    clip_bbox,
    filter_by_size,
    offset2bbox
)



__all__ = [
    'DacRoiPredictorPre', 'DacRoiPredictor'
]

@ROI_PREDICTOR_REGISTRY.register('dac_retina_pretrain')
class DacRoiPredictorPre(object):
    """Predictor for the first stage
    """

    def __init__(self,
                 pre_nms_score_thresh,
                 pre_nms_top_n,
                 post_nms_top_n,
                 roi_min_size,
                 merger=None,
                 nms=None,
                 clip_box=True,
                 gt_encode=True):
        self.pre_nms_score_thresh = pre_nms_score_thresh
        # self.apply_score_thresh_above = apply_score_thresh_above
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_cfg = nms
        self.roi_min_size = roi_min_size
        self.clip_box = clip_box
        if merger is not None:
            self.merger = build_merger(merger)
        else:
            self.merger = None
        self.gt_encode = gt_encode

    @torch.no_grad()
    @to_float32
    def predict(self, mlvl_anchors, mlvl_preds, image_info, return_pos_inds=None, strides=None):
        mlvl_resutls = []
        num_points = 0
        for lvl, (anchors, preds) in enumerate(zip(mlvl_anchors, mlvl_preds)):
            self.lvl = lvl
            results = self.single_level_predict(anchors, preds, image_info, num_points, return_pos_inds, strides=strides)
            num_points += len(anchors)
            mlvl_resutls.append(results)
        if len(mlvl_resutls) > 0:
            results = torch.cat(mlvl_resutls, dim=0)
        else:
            results = mlvl_anchors[0].new_zeros((1, 8)) if return_pos_inds else mlvl_anchors[0].new_zeros((1, 7))
        if self.merger is not None:
            results = self.merger.merge(results)
        if return_pos_inds:
            results, pos_inds = torch.split(results, [7, 1], dim=1)
            return {'dt_bboxes': results, 'pos_inds': pos_inds}
        return {'dt_bboxes': results}

    def regression(self, anchors, preds, image_info, strides=None):
        cls_pred, loc_pred = preds[:2]
        B, K = cls_pred.shape[:2]
        concat_anchors = torch.stack([anchors.clone() for _ in range(B)])
        if self.gt_encode:
            rois = offset2bbox(concat_anchors.view(B * K, 4), loc_pred.view(B * K, 4)).view(B, K, 4)  # noqa
        else:
            rois = self.fcos_offset2bbox(concat_anchors.view(B * K, 4), loc_pred.view(B * K, 4)).view(B, K, 4)
        return rois
    
    def fcos_offset2bbox(self, anchors, offset):
        cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
        cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
        locations = torch.cat([cx.view(-1, 1), cy.view(-1, 1)], dim=1)
        x1 = locations[:, 0] - offset[:, 0]
        y1 = locations[:, 1] - offset[:, 1]
        x2 = locations[:, 0] + offset[:, 2]
        y2 = locations[:, 1] + offset[:, 3]
        return torch.stack([x1, y1, x2, y2], -1)

    def single_level_predict(self, anchors, preds, image_info, num_points, return_pos_inds=None, strides=None):
        """
        Arguments:
            - anchors (FloatTensor, fp32): [K, 4]
            - preds[0] (cls_pred, FloatTensor, fp32): [B, K, C], C[i] -> class i+1, background class is excluded
            - preds[1] (loc_pred, FloatTensor, fp32): [B, K, 4]
            - image_info (list of FloatTensor): [B, >=2] (image_h, image_w, ...)

        Returns:
            rois (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
        """

        rois = self.regression(anchors, preds, image_info, strides=strides)

        cls_pred = preds[0]
        B, K, C = cls_pred.shape
        roi_min_size = self.roi_min_size
        pre_nms_top_n = self.pre_nms_top_n
        pre_nms_top_n = pre_nms_top_n if pre_nms_top_n > 0 else K
        post_nms_top_n = self.post_nms_top_n
        # if featmap size is too large, filter by score thresh to reduce computation
        if K > 120:
            score_thresh = self.pre_nms_score_thresh
        else:
            score_thresh = 0

        batch_rois = []
        for b_ix in range(B):
            # clip rois and filter rois which are too small
            image_rois = rois[b_ix]
            image_inds = torch.arange(K, dtype=torch.int64, device=image_rois.device)
            if self.clip_box:
                image_rois = clip_bbox(image_rois, image_info[b_ix])
            image_rois, filter_inds = filter_by_size(image_rois, roi_min_size)
            image_cls_pred = cls_pred[b_ix][filter_inds]
            image_inds = image_inds[filter_inds]
            if image_rois.numel() == 0:
                continue  # noqa E701

            for cls in range(C):
                cls_rois = image_rois
                scores = image_cls_pred[:, cls]
                cls_inds = image_inds
                assert not torch.isnan(scores).any()
                if score_thresh > 0:
                    # to reduce computation
                    keep_idx = torch.nonzero(scores > score_thresh).reshape(-1)
                    if keep_idx.numel() == 0:
                        continue  # noqa E701
                    cls_rois = cls_rois[keep_idx]
                    scores = scores[keep_idx]

                # do nms per image, only one class
                _pre_nms_top_n = min(pre_nms_top_n, scores.shape[0])
                scores, order = scores.topk(_pre_nms_top_n, sorted=True)
                cls_rois = cls_rois[order, :]
                cls_inds = cls_inds[order]
                if return_pos_inds:
                    cls_rois = torch.cat([cls_rois, scores[:, None], cls_inds[:, None]], dim=1)
                else:
                    cls_rois = torch.cat([cls_rois, scores[:, None]], dim=1)

                if self.nms_cfg is not None:
                    cls_rois, keep_idx = nms(cls_rois, self.nms_cfg)
                if post_nms_top_n > 0:
                    cls_rois = cls_rois[:post_nms_top_n]

                ix = cls_rois.new_full((cls_rois.shape[0], 1), b_ix)
                c = cls_rois.new_full((cls_rois.shape[0], 1), cls + 1)
                if return_pos_inds:
                    cls_rois, cls_inds = torch.split(cls_rois, [5, 1], dim=1)
                    cls_rois = torch.cat([ix, cls_rois, c, cls_inds], dim=1)
                else:
                    cls_rois = torch.cat([ix, cls_rois, c], dim=1)
                batch_rois.append(cls_rois)

        if len(batch_rois) == 0:
            return anchors.new_zeros((1, 8)) if return_pos_inds else anchors.new_zeros((1, 7))
        results = torch.cat(batch_rois, dim=0)
        # if return_pos_inds:
        #     results[:, 7].add_(num_points)
        return results


@ROI_PREDICTOR_REGISTRY.register('dac_retina')
class DacRoiPredictor(object):
    """Predictor for the first stage
    """

    def __init__(self,
                 pre_nms_score_thresh,
                 pre_nms_top_n,
                 post_nms_top_n,
                 roi_min_size,
                 merger=None,
                 nms=None,
                 clip_box=True,
                 gt_encode=True):
        self.pre_nms_score_thresh = pre_nms_score_thresh
        # self.apply_score_thresh_above = apply_score_thresh_above
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_cfg = nms
        self.roi_min_size = roi_min_size
        self.clip_box = clip_box
        if merger is not None:
            self.merger = build_merger(merger)
        else:
            self.merger = None
        self.gt_encode = gt_encode

    @torch.no_grad()
    @to_float32
    def predict(self, mlvl_anchors, mlvl_preds, image_info, return_pos_inds=None, strides=None):
        mlvl_resutls = []
        num_points = 0
        for lvl, (anchors, preds) in enumerate(zip(mlvl_anchors, mlvl_preds)):
            self.lvl = lvl
            results = self.single_level_predict(anchors, preds, image_info, num_points, return_pos_inds, strides=strides)
            num_points += len(anchors)
            mlvl_resutls.append(results)
        if len(mlvl_resutls) > 0:
            results = torch.cat(mlvl_resutls, dim=0)
        else:
            results = mlvl_anchors[0].new_zeros((1, 8)) if return_pos_inds else mlvl_anchors[0].new_zeros((1, 7))
        # if self.merger is not None:
        #     results = self.merger.merge(results)
        if return_pos_inds:
            results, pos_inds = torch.split(results, [7, 1], dim=1)
            return {'dt_bboxes': results, 'pos_inds': pos_inds}
        return {'dt_bboxes': results}

    def regression(self, anchors, preds, image_info, strides=None):
        cls_pred, loc_pred = preds[:2]
        B, K = cls_pred.shape[:2]
        concat_anchors = torch.stack([anchors.clone() for _ in range(B)])
        if self.gt_encode:
            rois = offset2bbox(concat_anchors.view(B * K, 4), loc_pred.view(B * K, 4)).view(B, K, 4)  # noqa
        else:
            # loc_pred = loc_pred * strides[0]
            rois = self.fcos_offset2bbox(concat_anchors.view(B * K, 4), loc_pred.view(B * K, 4)).view(B, K, 4) 
        return rois

    def fcos_offset2bbox(self, anchors, offset):
        cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
        cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
        locations = torch.cat([cx.view(-1, 1), cy.view(-1, 1)], dim=1)
        x1 = locations[:, 0] - offset[:, 0]
        y1 = locations[:, 1] - offset[:, 1]
        x2 = locations[:, 0] + offset[:, 2]
        y2 = locations[:, 1] + offset[:, 3]
        return torch.stack([x1, y1, x2, y2], -1)

    def single_level_predict(self, anchors, preds, image_info, num_points, return_pos_inds=None, strides=None):
        """
        Arguments:
            - anchors (FloatTensor, fp32): [K, 4]
            - preds[0] (cls_pred, FloatTensor, fp32): [B, K, C], C[i] -> class i+1, background class is excluded
            - preds[1] (loc_pred, FloatTensor, fp32): [B, K, 4]
            - image_info (list of FloatTensor): [B, >=2] (image_h, image_w, ...)

        Returns:
            rois (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
        """

        rois = self.regression(anchors, preds, image_info, strides=strides)
        # import numpy as np
        # np.save('dir_debug/rois.npy', rois.cpu().detach().numpy())

        cls_pred = preds[0]
        B, K, C = cls_pred.shape
        roi_min_size = self.roi_min_size
        # pre_nms_top_n = self.pre_nms_top_n
        # pre_nms_top_n = pre_nms_top_n if pre_nms_top_n > 0 else K
        # post_nms_top_n = self.post_nms_top_n
        # if featmap size is too large, filter by score thresh to reduce computation
        # if K > 120:
        #     score_thresh = self.pre_nms_score_thresh
        # else:
        #     score_thresh = 0

        batch_rois = []
        for b_ix in range(B):
            # clip rois and filter rois which are too small
            image_rois = rois[b_ix]
            image_inds = torch.arange(K, dtype=torch.int64, device=image_rois.device)
            if self.clip_box:
                image_rois = clip_bbox(image_rois, image_info[b_ix])
            image_rois, filter_inds = filter_by_size(image_rois, roi_min_size)
            image_cls_pred = cls_pred[b_ix][filter_inds]
            image_inds = image_inds[filter_inds]
            if image_rois.numel() == 0:
                continue  # noqa E701
            image_cls_pred = image_cls_pred.view(-1)
            index = torch.argmax(image_cls_pred)
            res_roi = torch.cat([image_rois[index].view(-1, 4), image_cls_pred[index].view(-1, 1)], dim=1)
            batch_rois.append(res_roi)

        if len(batch_rois) == 0:
            return anchors.new_zeros((1, 8)) if return_pos_inds else anchors.new_zeros((1, 7))
        results = torch.cat(batch_rois, dim=0)
        # np.save('dir_debug/final_roi.npy', results.cpu().detach().numpy())
        # if return_pos_inds:
        #     results[:, 7].add_(num_points)
        return results

def build_merger(merger_cfg):
    return ROI_MERGER_REGISTRY.build(merger_cfg)