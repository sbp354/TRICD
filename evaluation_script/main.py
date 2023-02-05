import json
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import contextlib
from sklearn.metrics import f1_score

from typing import Any, Dict, Optional, Tuple





from box_utils import box_iou, generalized_box_iou, obj_to_box, region_to_box, BoxFormat, BoxList
from grounding_recall import RecallTracker, get_recall, get_group_recall
from grounding_AP import PDEval, Params, get_AP

def grounding_eval(gt_file, pred_data):
    with open(gt_file) as f:
        gt_data = json.load(f)['grounding']
    
    with open("gt_grounding_annotations.json", 'w') as f:
        json.dump(gt_data, f)
    
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            gt_coco = COCO("gt_grounding_annotations.json")

    recall1 = get_recall(gt_data, pred_data)
    group_recall1 = get_group_recall(gt_data, pred_data)
    AP = get_AP(gt_data, gt_coco, pred_data)

    grounding_results = {
                        "all": {
                            "Recall@1": recall1['all'],
                            "GroupRecall@1": group_recall1['all'],
                            "AP": AP['all']
                        },

                        "coco_obj": {
                            "Recall@1": recall1['coco_obj'],
                            "GroupRecall@1": group_recall1['coco_obj'],
                            "AP": AP['coco_obj']
                        },
                        
                        "coco_rel": {
                            "Recall@1": recall1['coco_rel'],
                            "GroupRecall@1": group_recall1['coco_rel'],
                            "AP": AP['coco_rel']
                        },

                        "winoground": {
                            "Recall@1": recall1["winoground"],
                            "GroupRecall@1": group_recall1["winoground"],
                            "AP": AP["winoground"]
                        }

                        }
    return grounding_results

def VQA_eval(gt_file, pred_data):
    with open(gt_file) as f:
        gt_data = json.load(f)['vqa']
    
    image_ids = [a['image_id'] for a in gt_data['annotations']]
    gt_answers = [a['answer'] for a in gt_data['annotations']]
    pred_answers = [pred_data[str(i)] for i in image_ids]

    results = {}
    results['all'] = f1_score(gt_answers, pred_answers, average = 'macro')

    #Subgroups
    id2source = {a['image_id']: a['source'] if a['source']=='winoground' else a['coco_type'] for a in gt_data['annotations']}
    ## Winoground
    win_img_ids = [k for k, v in id2source.items() if v=='winoground']
    win_gt_answers = [a['answer'] for a in gt_data['annotations'] if a['image_id'] in(win_img_ids)]
    win_pred_answers = [pred_data[str(i)] for i in win_img_ids]
    results['winoground'] = f1_score(win_gt_answers, win_pred_answers, average = 'macro')

    ## COCO_obj
    coco_obj_img_ids = [k for k, v in id2source.items() if v=='object']
    coco_obj_gt_answers = [a['answer'] for a in gt_data['annotations'] if a['image_id'] in(coco_obj_img_ids)]
    coco_obj_pred_answers = [pred_data[str(i)] for i in coco_obj_img_ids]
    results['coco_obj'] = f1_score(coco_obj_gt_answers, coco_obj_pred_answers, average = 'macro')

    ## COCO_rel
    coco_rel_img_ids = [k for k, v in id2source.items() if v=='relation']
    coco_rel_gt_answers = [a['answer'] for a in gt_data['annotations'] if a['image_id'] in(coco_rel_img_ids)]
    coco_rel_pred_answers = [pred_data[str(i)] for i in coco_rel_img_ids]
    results['coco_rel'] = f1_score(coco_rel_gt_answers, coco_rel_pred_answers, average = 'macro')


    return results


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:
        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made
        `**kwargs`: keyword arguments that contains additional submission
    """
    output = {}
    output["result"] = {}

    #Load user submission file to see which tasks we need to evaluate (grounding, VQA or both)
    with open(user_submission_file) as f:
        pred_data = json.load(f)

    if 'grounding' in pred_data:
        #Grounding metrics are Recall@1, GroupRecall@1 and AP
        output['result']['grounding'] = grounding_eval(test_annotation_file, pred_data['grounding'])
    
    if 'vqa' in pred_data:
        output['result']['vqa'] = VQA_eval(test_annotation_file, pred_data['vqa'])

    
    # To display the results in the result file
    output["submission_result"] = output["result"]
    return output
