import os
from config import load_config
from dataset import build_dataloader
from evaluator import build_evaluator
from utils import *
import pickle as pkl
import pandas as pd
import json
import ipdb
from collections import Counter


def predictions_to_json(verb_scores, noun_scores, action_scores, action_ids, a_to_vn, top_actions=100, version='0.1', sls=None):
    """Save verb, noun and action predictions to json for submitting them to the EPIC-Kitchens leaderboard"""
    predictions = {'version': version,
                   'challenge': 'action_anticipation', 'results': {}}

    if sls is not None:
        predictions['sls_pt'] = 1
        predictions['sls_tl'] = 4
        predictions['sls_td'] = 3

    row_idxs = np.argsort(action_scores)[:, ::-1]
    top_100_idxs = row_idxs[:, :top_actions]

    action_scores = action_scores[np.arange(
        len(action_scores)).reshape(-1, 1), top_100_idxs]

    for i, v, n, a, ai in zip(action_ids, verb_scores, noun_scores, action_scores, top_100_idxs):
        predictions['results'][str(i)] = {}
        predictions['results'][str(i)]['verb'] = {str(
            ii): float(vv) for ii, vv in enumerate(v)}
        predictions['results'][str(i)]['noun'] = {str(
            ii): float(nn) for ii, nn in enumerate(n)}
        predictions['results'][str(i)]['action'] = {
            "%d,%d" % a_to_vn[ii]: float(aa) for ii, aa in zip(ai, a)}
    
    return predictions


def ensemble_scores(resume_paths, dataset_split, weight_list):
    all_gt = 0
    all_pred_verb = 0
    all_pred_noun = 0
    all_pred_action = 0

    for idx in range(len(resume_paths)):
        resume_path = resume_paths[idx]
        result_path = os.path.join(resume_path, 'best.pt.' + dataset_split + '.pkl')
        with open(result_path, 'rb') as f:
            all_data = pkl.load(f)
        
        all_pred_verb = all_pred_verb + all_data['pred']['verb'] * weight_list[idx]
        all_pred_noun = all_pred_noun + all_data['pred']['noun'] * weight_list[idx]
        all_pred_action = all_pred_action + all_data['pred']['action'] * weight_list[idx]
        
        all_gt = all_data['gt']

    all_pred = {}
    all_pred['verb'] = all_pred_verb
    all_pred['noun'] = all_pred_noun
    all_pred['action'] = all_pred_action
    
    return all_pred, all_gt


def main(config):
    logger, _ = build_logger(path=None, console=True, tensorboard_log=False)
    logger.info(config)

    eval_loader = build_dataloader(logger, config.eval.data, shuffle=False, ddp=False)
    evaluator = build_evaluator(eval_loader.dataset)

    dataset_split = config.eval.data.split # test test test_seen test_unseen
    dataset = config.eval.data.name
    '''
    ## 分数据集以及划分
    weight_list = [1, 1, 1]
    if dataset == 'EPIC-KITCHENS-100':
        ## ek100 验证
        if dataset_split == 'valid':
            rgbtsn_timestamps = ['Jul29_03-57-31', 'Aug27_10-55-06', 'Aug27_11-32-29', 'Aug27_14-30-02', 'Aug28_03-38-34', 'Aug27_21-54-51', 'Aug28_07-31-23', 'Aug28_14-14-59', 'Aug27_08-16-25', 'Aug27_14-52-05', 'Aug27_17-23-15', 'Aug27_18-00-28', 'Aug28_09-59-54', 'Aug28_01-11-00', 'Aug28_00-53-43', 'Aug28_06-47-25']
            rgbtsm_timestamps = ['Aug27_09-52-57', 'Aug07_05-53-45', 'Aug26_11-42-02', 'Aug27_03-59-16', 'Aug30_15-39-41', 'Aug30_18-59-39', 'Aug29_13-23-12']
            obj_timestamps = ['Sep03_13-47-22', 'Sep04_05-39-54','Sep04_14-00-32', 'Sep04_16-30-47', 'Sep04_21-02-05', 'Sep04_22-22-13', 'Sep04_23-28-11', 'Sep04_23-57-22', 'Sep05_01-17-02', 'Sep05_07-06-45', 'Sep05_08-31-38']
            #obj_timestamps = ['Aug05_02-41-27', 'Sep02_13-20-47', 'Sep02_13-21-46', 'Sep02_19-40-09', 'Sep02_22-54-18']

            all_pred_results = {}
            for rgbtsn_timestamp in rgbtsn_timestamps:
                for rgbtsm_timestamp in rgbtsm_timestamps:
                    for obj_timestamp in obj_timestamps:
                        resume_paths = ['./exp/EK100RGBTSN/train/' + rgbtsn_timestamp, './exp/EK100RGBTSM/train/' + rgbtsm_timestamp, './exp/EK100OBJFRCNN/train/' + obj_timestamp]
                        ## get ensemble scores
                        all_pred, all_gt = ensemble_scores(resume_paths, dataset_split, weight_list)
                        result = evaluator(all_pred, all_gt)

                        curr_key = 'rgb_tsn: ' + rgbtsn_timestamp + ', rgb_tsm:' + rgbtsm_timestamp + ', obj: '+ obj_timestamp
                        all_pred_results[curr_key] = result
                        #print(result)
    
            new_all_pred_results = {}
            for key, value in all_pred_results.items():
                if (value['All_A'] > 0.183) and (value['Uns_V'] > 0.318) and (value['Uns_A'] > 0.147) and (value['Tail_A'] > 0.158):
                    new_all_pred_results[key] = value

            with open('./ensemble_results.json', 'w') as f:
                f.write(json.dumps(new_all_pred_results))

    import ipdb; ipdb.set_trace()
    '''
    ## 分数据集以及划分
    if dataset == 'EPIC-KITCHENS-100':
        ## ek100 验证
        if dataset_split == 'valid':
            resume_paths = ['./exp/EK100RGBTSN/train/Aug27_11-32-29', './exp/EK100RGBTSM/train/Aug30_15-39-41', './exp/EK100OBJFRCNN/train/Sep02_19-40-09']
            weight_list = [1, 1, 1]
        elif dataset_split == 'test':
            resume_paths = ['./exp/EK100RGBTSN/train/Aug28_00-53-43', './exp/EK100RGBTSM/train/Aug30_15-39-41', './exp/EK100OBJFRCNN/train/Sep04_21-02-05']
            weight_list = [1, 1, 1]
    elif dataset == 'EPIC-KITCHENS-55':
        if dataset_split == 'valid':
            resume_paths = ['./exp/EK55RGBTSN/train/Aug15_10-54-39', './exp/EK55RGBTSM/train/Aug12_16-51-11', './exp/EK55RGBCSN/train/Aug14_22-19-47', './exp/EK55OBJFRCNN/train/Aug12_18-16-36']
            weight_list = [0.5, 1, 1, 0.5]
        elif dataset_split == 'test_seen':
            resume_paths = ['./exp/EK55RGBTSM/train/Aug23_07-58-51']
            weight_list = [1]
        elif dataset_split == 'test_unseen':
            resume_paths = ['./exp/EK55RGBTSM/train/Aug23_07-58-51']
            weight_list = [1]
    elif dataset == 'EGTEA_GAZE+':
        if dataset_split == 'valid1':
            resume_paths = ['./exp/EGATEGAZERGB1/train/Aug09_20-06-38', './exp/EGATEGAZEFLOW1/train/Aug09_18-10-48'] # Aug08_14-41-50  Aug08_14-42-29 Aug08_14-50-52
            weight_list = [1, 0.005]
        elif dataset_split == 'valid2':
            resume_paths = ['./exp/EGATEGAZERGB2/train/Aug09_18-27-01', './exp/EGATEGAZEFLOW2/train/Aug09_21-39-50']
            weight_list = [1, 0.005]
        elif dataset_split == 'valid3':
            resume_paths = ['./exp/EGATEGAZERGB3/train/Aug09_21-51-52', './exp/EGATEGAZEFLOW3/train/Aug09_19-54-03']
            weight_list = [1, 0.005]
    
    ## get ensemble scores
    all_pred, all_gt = ensemble_scores(resume_paths, dataset_split, weight_list)
    
    ## split
    if not 'test' in dataset_split:
        result = evaluator(all_pred, all_gt)
        print(result)
    else:
        ## get dcr ensemble test results
        if dataset == 'EPIC-KITCHENS-100':
            dataset_root_path = './data/EK100/EK100_action_composition.json'
        elif dataset == 'EPIC-KITCHENS-55':
            dataset_root_path = './data/EK55/EK55_action_composition.json'  
        
        with open(dataset_root_path, 'rb') as f:
            action_infos = json.load(f)
        a_to_vn = {action_idx: tuple(action_infos[action_idx]) for action_idx in range(len(action_infos))}
       
        ensemble_preds = predictions_to_json(all_pred['verb'], all_pred['noun'], all_pred['action'], all_gt['id'], a_to_vn, version = '0.2' if dataset == 'EPIC-KITCHENS-100' else '0.1', sls=True)

        with open(os.path.join('./' + dataset + '_' + dataset_split + '.json'), 'w') as f:
            f.write(json.dumps(ensemble_preds, indent=4, separators=(',',': ')))


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    initialize(config)
    main(config)
