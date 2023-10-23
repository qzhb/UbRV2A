"""Main training/test program for RULSTM"""
from argparse import ArgumentParser
from dataset import SequenceDataset
from os.path import join
from datetime import datetime
from models import RULSTM, RULSTMFusion
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utils import topk_accuracy, ValueMeter, topk_accuracy_multiple_timesteps, get_marginal_indexes, marginalize, softmax,  topk_recall_multiple_timesteps, tta, predictions_to_json, MeanTopKRecallMeter
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import os
import math
import random
import pickle
pd.options.display.float_format = '{:05.2f}'.format


parser = ArgumentParser(description="Training program for RULSTM")
parser.add_argument('mode', type=str, choices=['train', 'validate', 'test', 'test', 'validate_json'], default='train',
                    help="Whether to perform training, validation or test.\
                            If test is selected, --json_directory must be used to provide\
                            a directory in which to save the generated jsons.")
parser.add_argument('path_to_data', type=str,
                    help="Path to the data folder, \
                            containing all LMDB datasets")
parser.add_argument('path_to_models', type=str,
                    help="Path to the directory where to save all models")
parser.add_argument('--alpha', type=float, default=0.25,
                    help="Distance between time-steps in seconds")
parser.add_argument('--S_enc', type=int, default=6,
                    help="Number of encoding steps. \
                            If early recognition is performed, \
                            this value is discarded.")
parser.add_argument('--S_ant', type=int, default=8,
                    help="Number of anticipation steps. \
                            If early recognition is performed, \
                            this is the number of frames sampled for each action.")
parser.add_argument('--task', type=str, default='anticipation', choices=[
                    'anticipation', 'early_recognition'], help='Task to tackle: \
                            anticipation or early recognition')
parser.add_argument('--img_tmpl', type=str,
                    default='frame_{:010d}.jpg', help='Template to use to load the representation of a given frame')
parser.add_argument('--modality', type=str, default='rgb',
                    choices=['rgb', 'flow', 'obj', 'fusion'], help = "Modality. Rgb/flow/obj represent single branches, whereas fusion indicates the whole model with modality attention.")
parser.add_argument('--sequence_completion', action='store_true',
                    help='A flag to selec sequence completion pretraining rather than standard training.\
                            If not selected, a valid checkpoint for sequence completion pretraining\
                            should be available unless --ignore_checkpoints is specified')
parser.add_argument('--mt5r', action='store_true')


## curriculum learning
parser.add_argument('--uncertainty_learning', action='store_true', help='Whether to use training')

parser.add_argument('--simple_fusion', action='store_true')

parser.add_argument('--curriculm_learning', action='store_true', help='Whether to use training')
parser.add_argument('--cl_start_epoch', type=int, default=10, help="Training epochs")
parser.add_argument('--cl_end_epoch', type=int, default=10, help="Training epochs")
parser.add_argument('--curriculum_method', default='linear', type=str, choices=['piecewise_linear', 'linear', 'log', 'quad', 'root', 'exp', 'geom'], help='curriculum mnethod')
parser.add_argument('--ratio_range', nargs='+', type=float, help="")
parser.add_argument('--curriculum_episode_num', type=int, default=5, help='curriculum episode nums which can be used for generate curriculum')

parser.add_argument('--no_scale', action='store_true', help='')
parser.add_argument('--direct_scale', action='store_true', help='')
parser.add_argument('--sigmoid_scale', action='store_true', help='')
parser.add_argument('--mean_direct_scale', action='store_true', help='')
parser.add_argument('--mean_sigmoid_scale', action='store_true', help='')

parser.add_argument('--nomixup', action='store_true', help='')
parser.add_argument('--ranking_loss', action='store_true', help='')
parser.add_argument('--wd_loss', action='store_true', help='')
parser.add_argument('--expand_targets', action='store_true', help='')
parser.add_argument('--smooth_value', type=float, default=0.4, help="")
parser.add_argument('--noise_percent', type=float, default=0.0, help="")

parser.add_argument('--ranking_loss_weight', type=float, default=0.1, help="")
parser.add_argument('--wd_loss_weight', type=float, default=1e-4, help="")

parser.add_argument('--resume_timestamp',  nargs='+', default='', type=str, help='resume timestamp')

## added for resubmit
parser.add_argument('--mcdropout', action='store_true')

## model
parser.add_argument('--num_class', type=int, default=2513,
                    help='Number of classes')
parser.add_argument('--hidden', type=int, default=1024,
                    help='Number of hidden units')
parser.add_argument('--feat_in', type=int, default=1024,
                    help='Input size. If fusion, it is discarded (see --feats_in)')
parser.add_argument('--feats_in', type=int, nargs='+', default=[1024, 1024, 352],
                    help='Input sizes when the fusion modality is selected.')
parser.add_argument('--dropout', type=float, default=0.8, help="Dropout rate")

parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
parser.add_argument('--num_workers', type=int, default=0,
                    help="Number of parallel thread to fetch the data")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")

parser.add_argument('--display_every', type=int, default=10,
                    help="Display every n iterations")
parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
parser.add_argument('--visdom', action='store_true',
                    help="Whether to log using visdom")

parser.add_argument('--ignore_checkpoints', action='store_true',
                    help='If specified, avoid loading existing models (no pre-training)')
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume suspended training')

parser.add_argument('--ek100', action='store_true',
                    help="Whether to use EPIC-KITCHENS-100")
parser.add_argument('--pretrained_sc_path',  type=str)

parser.add_argument('--egategazeplus', action='store_true',
                    help="Whether to use EGATEGazePlus")
parser.add_argument('--split_num', type=int, default=1)

parser.add_argument('--json_directory', type=str, default = None, help = 'Directory in which to save the generated jsons.')

args = parser.parse_args()

if args.mode == 'test' or args.mode=='validate_json':
    assert args.json_directory is not None

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.task == 'anticipation':
    exp_name = f"RULSTM-{args.task}_{args.alpha}_{args.S_enc}_{args.S_ant}_{args.modality}"
else:
    exp_name = f"RULSTM-{args.task}_{args.alpha}_{args.S_ant}_{args.modality}"

if args.mt5r:
    exp_name += '_mt5r'

if args.sequence_completion:
    exp_name += '_sequence_completion'

args.timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
args.save_path = join(args.path_to_models, args.timestamp)
os.makedirs(args.save_path)

if args.visdom:
    # if visdom is required
    # load visdom loggers from torchent
    from torchnet.logger import VisdomPlotLogger, VisdomSaver
    # define loss and accuracy logger
    visdom_loss_logger = VisdomPlotLogger('line', env=exp_name, opts={
                                          'title': 'Loss', 'legend': ['training', 'validation']})
    visdom_accuracy_logger = VisdomPlotLogger('line', env=exp_name, opts={
                                              'title': 'Top5 Acc@1s', 'legend': ['training', 'validation']})
    # define a visdom saver to save the plots
    visdom_saver = VisdomSaver(envs=[exp_name])

def get_loader(mode, override_modality = None):
    if override_modality:
        path_to_lmdb = join(args.path_to_data, override_modality)
    else:
        if args.egategazeplus:
            path_to_lmdb = join(args.path_to_data, args.modality + '_s' + str(args.split_num)) if args.modality != 'fusion' else [join(args.path_to_data, m + '_s' + str(args.split_num)) for m in ['rgb', 'flow']]
        else:
            path_to_lmdb = join(args.path_to_data, args.modality) if args.modality != 'fusion' else [join(args.path_to_data, m) for m in ['rgb', 'flow', 'obj']]

    if args.ek100:
        kargs = {
            'path_to_lmdb': path_to_lmdb,
            'path_to_csv': join(args.path_to_data, f"{mode}.csv"),
            'time_step': args.alpha,
            'img_tmpl': args.img_tmpl,
            'action_samples': args.S_ant if args.task == 'early_recognition' else None,
            'past_features': args.task == 'anticipation',
            'sequence_length': args.S_enc + args.S_ant,
            'label_type': ['verb', 'noun', 'action'] if args.mode != 'train' else 'action',
            'challenge': 'test' in mode,
            'ek100': True,
            'noise_percent': args.noise_percent,
            'mode': mode
        }
    else:
        if args.egategazeplus:
            kargs = {
                'path_to_lmdb': path_to_lmdb,
                'path_to_csv': join(args.path_to_data, f"{mode}{args.split_num}.csv"),
                'time_step': args.alpha,
                'img_tmpl': args.img_tmpl,
                'action_samples': args.S_ant,
                'past_features': args.task == 'anticipation',
                'sequence_length': args.S_enc + args.S_ant,
                'label_type': ['verb', 'noun', 'action'] if args.mode != 'train' else 'action',
                'challenge': 'test' in mode,
                'egategazeplus': True,
                'split_num': args.split_num,
                'noise_percent': args.noise_percent,
                'mode': mode
            }
        else:
            kargs = {
            'path_to_lmdb': path_to_lmdb,
            'path_to_csv': join(args.path_to_data, f"{mode}.csv"),
            'time_step': args.alpha,
            'img_tmpl': args.img_tmpl,
            'action_samples': args.S_ant if args.task == 'early_recognition' else None,
            'past_features': args.task == 'anticipation',
            'sequence_length': args.S_enc + args.S_ant,
            'label_type': ['verb', 'noun', 'action'] if args.mode != 'train' else 'action',
            'challenge': 'test' in mode,
            'noise_percent': args.noise_percent,
            'mode': mode
            }

    _set = SequenceDataset(**kargs)

    return _set

def get_model():
    if args.modality != 'fusion':  # single branch
        model = RULSTM(args.num_class, args.feat_in, args.hidden,
                       args.dropout, sequence_completion=args.sequence_completion)
        # load checkpoint only if not in sequence completion mode
        # and inf the flag --ignore_checkpoints has not been specified

        if args.mode == 'train' and not args.ignore_checkpoints and not args.sequence_completion:
            
            if args.egategazeplus:
                checkpoint = torch.load(join(args.path_to_models, args.pretrained_sc_path, exp_name + '_sequence_completion_best.pth.tar'))['state_dict']
            else:
                checkpoint = torch.load(join(args.path_to_models, exp_name + '_sequence_completion_best.pth.tar'))['state_dict']

            model_dict = model.state_dict()
            pretrained_dict = {key: value for key, value in checkpoint.items() if key in model_dict.keys()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    else:
        if args.egategazeplus:
            if args.simple_fusion:
                rgb_model = RULSTM(args.num_class, args.feats_in[0], args.hidden, args.dropout)
                flow_model = RULSTM(args.num_class, args.feats_in[1], args.hidden, args.dropout)
            else:
                rgb_model = RULSTM(args.num_class, args.feats_in[0], args.hidden, args.dropout, return_context = args.task=='anticipation')
                flow_model = RULSTM(args.num_class, args.feats_in[1], args.hidden, args.dropout, return_context = args.task=='anticipation')
                
            if args.task=='early_recognition' or (args.mode == 'train' and not args.ignore_checkpoints) or args.simple_fusion:
                checkpoint_rgb = torch.load(join(args.path_to_models, args.resume_timestamp[0], \
                        exp_name.replace('fusion','rgb') +'_best.pth.tar'))['state_dict']
                checkpoint_flow = torch.load(join(args.path_to_models, args.resume_timestamp[1], \
                        exp_name.replace('fusion','flow') +'_best.pth.tar'))['state_dict']
                
                rgb_model.load_state_dict(checkpoint_rgb)
                flow_model.load_state_dict(checkpoint_flow)
                
            if args.task == 'early_recognition' or args.simple_fusion:
                return [rgb_model, flow_model]

            model = RULSTMFusion([rgb_model, flow_model], args.hidden, args.dropout)

        else:
            if args.simple_fusion:
                rgb_model = RULSTM(args.num_class, args.feats_in[0], args.hidden, args.dropout)
                flow_model = RULSTM(args.num_class, args.feats_in[1], args.hidden, args.dropout)
                obj_model = RULSTM(args.num_class, args.feats_in[2], args.hidden, args.dropout)
            else:
                rgb_model = RULSTM(args.num_class, args.feats_in[0], args.hidden, args.dropout, return_context = args.task=='anticipation')
                flow_model = RULSTM(args.num_class, args.feats_in[1], args.hidden, args.dropout, return_context = args.task=='anticipation')
                obj_model = RULSTM(args.num_class, args.feats_in[2], args.hidden, args.dropout, return_context = args.task=='anticipation')

            if args.task=='early_recognition' or (args.mode == 'train' and not args.ignore_checkpoints) or args.simple_fusion:
                checkpoint_rgb = torch.load(join(args.path_to_models, args.resume_timestamp[0], \
                        exp_name.replace('fusion','rgb') +'_best.pth.tar'))['state_dict']
                checkpoint_flow = torch.load(join(args.path_to_models, args.resume_timestamp[1], \
                        exp_name.replace('fusion','flow') +'_best.pth.tar'))['state_dict']
                checkpoint_obj = torch.load(join(args.path_to_models, args.resume_timestamp[2], \
                        exp_name.replace('fusion','obj') +'_best.pth.tar'))['state_dict']

                rgb_model.load_state_dict(checkpoint_rgb)
                flow_model.load_state_dict(checkpoint_flow)
                obj_model.load_state_dict(checkpoint_obj)
            
            if args.task == 'early_recognition' or args.simple_fusion:
                return [rgb_model, flow_model, obj_model]

            model = RULSTMFusion([rgb_model, flow_model, obj_model], args.hidden, args.dropout)

    return model


def load_checkpoint(model, best=False):
    if best:
        chk = torch.load(join(args.path_to_models, args.resume_timestamp[0], exp_name + '_best.pth.tar'))
    else:
        chk = torch.load(join(args.path_to_models, args.resume_timestamp[0], exp_name + '.pth.tar'))

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf


def save_model(model, epoch, perf, best_perf, is_best=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'perf': perf, 'best_perf': best_perf}, join(args.save_path, exp_name + '.pth.tar'))
    
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            args.save_path, exp_name + '_best.pth.tar'))

    if args.visdom:
        # save visdom logs for persitency
        visdom_saver.save()


def log(mode, epoch, loss_meter, accuracy_meter, best_perf=None, green=False):
    if green:
        print('\033[92m', end="")

    print(
        f"[{mode}] Epoch: {epoch:0.2f}. "
        f"Loss: {loss_meter.value():.2f}. "
        f"Accuracy: {accuracy_meter.value():.2f}% ", end="")

    if best_perf:
        print(f"[best: {best_perf:0.2f}]%", end="")

    print('\033[0m')

    if args.visdom:
        visdom_loss_logger.log(epoch, loss_meter.value(), name=mode)
        visdom_accuracy_logger.log(epoch, accuracy_meter.value(), name=mode)


def get_scores_simple_fusion(models, loaders):
    verb_scores = 0
    noun_scores = 0
    action_scores = 0
    for model, loader in zip(models, loaders):
        outs = get_scores(model, loader)
        verb_scores += outs[0]
        noun_scores += outs[1]
        action_scores += outs[2]

    verb_scores /= len(models)
    noun_scores /= len(models)
    action_scores /= len(models)

    return [verb_scores, noun_scores, action_scores] + list(outs[3:])


def get_scores_mcdropout(model, loader, challenge=False, include_discarded = False):
    #model.eval()
    predictions = []
    labels = []
    ids = []

    visualization_results = {} 
    
    num_samples = 20
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x = batch['past_features' if args.task ==
                      'anticipation' else 'action_features']
            if type(x) == list:
                x = [xx.to(device) for xx in x]
            else:
                x = x.to(device)

            y = batch['label'].numpy()

            ids.append(batch['id'])

            prob_action_total = torch.zeros((num_samples, batch['past_features'].shape[0], 8, 2513))
            uncert_total = torch.zeros((num_samples, batch['past_features'].shape[0], 8, 1))

            for num_idx in range(num_samples):

                preds, predicted_uncertainties = model(x)
                preds = preds[:, -args.S_ant:, :].contiguous()
                predicted_uncertainties = predicted_uncertainties[:, -args.S_ant:, :].contiguous()
                predicted_uncertainties = predicted_uncertainties.mean(dim=2, keepdim=True)
                
                prob_action_total[num_idx] = preds
                uncert_total[num_idx] = predicted_uncertainties

            preds = torch.mean(prob_action_total, 0)
            predicted_uncertainties = torch.mean(uncert_total, 0)
            
            #import ipdb; ipdb.set_trace()
            ## ['id', 'index', 'frame_index', 'label', 'statistic_action_label', 'conceptnet_action_label', 'rgb']
            for idx in range(batch['label'].shape[0]):
                curr_id = int(batch['id'][idx])
                curr_index = int(batch['index'][idx])
                curr_label = int(batch['label'][idx][-1])
                curr_frame_index = []                    
                for f_idx in range(len(batch['frame_index'])):
                    curr_frame_index.append(batch['frame_index'][f_idx][idx])
                
                curr_preds = F.softmax(preds[idx, -4, :], dim=-1).cpu().numpy()
                curr_uncertainty_with_sigmoid = float(F.sigmoid(predicted_uncertainties[idx, -4]).cpu())
                curr_uncertainty = float(predicted_uncertainties[idx, -4].cpu())
                curr_preds_uncertainty = curr_preds / curr_uncertainty
                
                visualization_results[curr_id] = {
                    'id': curr_id,
                    'index': curr_index,
                    'label': curr_label,
                    'frame_index': curr_frame_index,
                    'pred': curr_preds,
                    'uncertainty': curr_uncertainty,
                    'pred_uncertainty': curr_preds_uncertainty
                }
            
            preds = preds / predicted_uncertainties
            
            preds = preds.cpu().numpy()

            predictions.append(preds)
            labels.append(y)
    
    with open('../uncertainty_visualization/rulstm_uncertainty_ek55_' + args.modality + '_visualization_results.pickle', 'wb') as f:
        pickle.dump(visualization_results, f)
    #import ipdb; ipdb.set_trace()
    
    action_scores = np.concatenate(predictions)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    if args.egategazeplus:
        actions = pd.read_csv(
            join(args.path_to_data, 'actions_for_validation.csv'), index_col='id')
    else:
        actions = pd.read_csv(
            join(args.path_to_data, 'actions.csv'), index_col='id')

    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')
   
    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))

    verb_scores = marginalize(action_probs, vi).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)
    noun_scores = marginalize(action_probs, ni).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)

    if include_discarded:
        dlab = np.array(loader.dataset.discarded_labels)
        dislab = np.array(loader.dataset.discarded_ids)
        ids = np.concatenate([ids, dislab])
        num_disc = len(dlab)
        labels = np.concatenate([labels, dlab])
        verb_scores = np.concatenate((verb_scores, np.zeros((num_disc, *verb_scores.shape[1:]))))
        noun_scores = np.concatenate((noun_scores, np.zeros((num_disc, *noun_scores.shape[1:]))))
        action_scores = np.concatenate((action_scores, np.zeros((num_disc, *action_scores.shape[1:]))))

    ubdcr_mcd_data_uncertainty = []
    ubdcr_mcd_model_uncertainty = []
    epsilon = 1e-10
    for key, value in visualization_results.items():  #['id', 'index', 'label', 'verb_label', 'noun_label', 'frame_index', 'pred', 'uncertainty']
        ubdcr_mcd_data_uncertainty.append(value['uncertainty'])
        
        #import ipdb; ipdb.set_trace()
        probs = value['pred_uncertainty'] + epsilon
        entropy = -np.sum(probs * np.log2(probs))
        ubdcr_mcd_model_uncertainty.append(entropy)

    ubdcr_mcd_data_uncertainty = np.array(ubdcr_mcd_data_uncertainty)
    ubdcr_mcd_model_uncertainty = np.array(ubdcr_mcd_model_uncertainty)
    print(ubdcr_mcd_data_uncertainty.mean())
    print(ubdcr_mcd_model_uncertainty.mean())
    
    if labels.max()>0 and not challenge:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2], ids
    else:
        return verb_scores, noun_scores, action_scores, ids


def get_scores(model, loader, challenge=False, include_discarded = False):
    model.eval()
    predictions = []
    labels = []
    ids = []

    visualization_results = {} 
    
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x = batch['past_features' if args.task ==
                      'anticipation' else 'action_features']
            if type(x) == list:
                x = [xx.to(device) for xx in x]
            else:
                x = x.to(device)

            y = batch['label'].numpy()

            ids.append(batch['id'])

            preds, predicted_uncertainties = model(x)
            preds = preds[:, -args.S_ant:, :].contiguous()
            predicted_uncertainties = predicted_uncertainties[:, -args.S_ant:, :].contiguous()

            predicted_uncertainties = predicted_uncertainties.mean(dim=2, keepdim=True)
            
            #import ipdb; ipdb.set_trace()
            ## ['id', 'index', 'frame_index', 'label', 'statistic_action_label', 'conceptnet_action_label', 'rgb']
            for idx in range(batch['label'].shape[0]):
                curr_id = int(batch['id'][idx])
                curr_index = int(batch['index'][idx])
                curr_label = int(batch['label'][idx][-1])
                curr_frame_index = []                    
                for f_idx in range(len(batch['frame_index'])):
                    curr_frame_index.append(batch['frame_index'][f_idx][idx])
                
                curr_preds = preds[idx, -4, :].cpu().numpy()
                curr_uncertainty_with_sigmoid = float(F.sigmoid(predicted_uncertainties[idx, -4]).cpu())
                curr_uncertainty = float(predicted_uncertainties[idx, -4].cpu())
                curr_preds_uncertainty = curr_preds / curr_uncertainty
                
                visualization_results[curr_id] = {
                    'id': curr_id,
                    'index': curr_index,
                    'label': curr_label,
                    'frame_index': curr_frame_index,
                    'pred': curr_preds,
                    'uncertainty': curr_uncertainty
                }
            
            preds = preds / predicted_uncertainties
            
            preds = preds.cpu().numpy()

            predictions.append(preds)
            labels.append(y)
    
    with open('../uncertainty_visualization/rulstm_uncertainty_ek55_' + args.modality + '_visualization_results.pickle', 'wb') as f:
        pickle.dump(visualization_results, f)
    #import ipdb; ipdb.set_trace()
    
    action_scores = np.concatenate(predictions)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    if args.egategazeplus:
        actions = pd.read_csv(
            join(args.path_to_data, 'actions_for_validation.csv'), index_col='id')
    else:
        actions = pd.read_csv(
            join(args.path_to_data, 'actions.csv'), index_col='id')

    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')
   
    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))

    verb_scores = marginalize(action_probs, vi).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)
    noun_scores = marginalize(action_probs, ni).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)

    if include_discarded:
        dlab = np.array(loader.dataset.discarded_labels)
        dislab = np.array(loader.dataset.discarded_ids)
        ids = np.concatenate([ids, dislab])
        num_disc = len(dlab)
        labels = np.concatenate([labels, dlab])
        verb_scores = np.concatenate((verb_scores, np.zeros((num_disc, *verb_scores.shape[1:]))))
        noun_scores = np.concatenate((noun_scores, np.zeros((num_disc, *noun_scores.shape[1:]))))
        action_scores = np.concatenate((action_scores, np.zeros((num_disc, *action_scores.shape[1:]))))

    ubdcr_mcd_data_uncertainty = []
    for key, value in visualization_results.items():  #['id', 'index', 'label', 'verb_label', 'noun_label', 'frame_index', 'pred', 'uncertainty']
        ubdcr_mcd_data_uncertainty.append(value['uncertainty'])
    
    ubdcr_mcd_data_uncertainty = np.array(ubdcr_mcd_data_uncertainty)
    print(ubdcr_mcd_data_uncertainty.mean())
   
    if labels.max()>0 and not challenge:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2], ids
    else:
        return verb_scores, noun_scores, action_scores, ids


def listMLE(y_pred, y_true, eps=1e-10):
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    _, indices = y_true_shuffled.sort(descending=False, dim=-1)

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

    ## listmle 化简后的等价形式    
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    return torch.mean(torch.sum(observation_loss, dim=1))


def trainval(model, optimizer, epochs, start_epoch, start_best_perf):

     #### get dataset and dataloader
    train_set = get_loader('training')
    validation_set = get_loader('validation')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    #### cl params
    n_samples = len(train_set)
    ce_loss_function = torch.nn.CrossEntropyLoss()
    
    break_flag = 99
    break_count = 0

    """Training/Validation code"""
    best_perf = start_best_perf  # to keep track of the best performing epoch
    for epoch in range(start_epoch, epochs):
        # define training and validation meters
        loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        if args.mt5r:
            accuracy_meter = {'training': MeanTopKRecallMeter(args.num_class), 'validation': MeanTopKRecallMeter(args.num_class)}
        else:
            accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}

        mode = 'training'
        # enable gradients only if training
        with torch.set_grad_enabled(True):
            model.train()
            
            for i, batch in enumerate(train_loader):
                x = batch['past_features' if args.task ==
                            'anticipation' else 'action_features']

                if type(x) == list:
                    x = [xx.to(device) for xx in x]
                else:
                    x = x.to(device)
                    
                y = batch['label'].to(device)

                bs = y.shape[0]  # batch size
                
                if args.uncertainty_learning and (not args.sequence_completion):
                    if args.modality == 'fusion':
                        ## anticipation: predicted_uncertainties [bs, len, C], others [bs]
                        action_preds, predicted_uncertainties = model(x, mode)

                        action_preds = action_preds[:, -args.S_ant:, :].contiguous()
                        predicted_uncertainties = predicted_uncertainties[:, -args.S_ant:, :].contiguous()

                        ## mixup uncertainty
                        predicted_uncertainties = predicted_uncertainties.mean(dim=2, keepdim=True)
                        scaled_action_preds = action_preds / predicted_uncertainties
                        linear_action_preds = scaled_action_preds.view(-1, scaled_action_preds.shape[-1])

                        # expand smooth label
                        statistic_action_label = batch['statistic_action_label'].to(device) # [bs, c]
                        statistic_action_label = (statistic_action_label > 0).int()
                        conceptnet_action_label = batch['conceptnet_action_label'].to(device)
                        conceptnet_action_label = (conceptnet_action_label > 0).int()
                        expanded_targets = statistic_action_label + conceptnet_action_label
                        
                        ## 将target_a 和 target_b 在smooth label中置零
                        y_T = y.unsqueeze(0).T # 转置
                        expanded_targets.scatter_(1, y_T, 0)
                        
                        # 计算smooth label的数量
                        smooth_label_nums = expanded_targets.sum(dim=1) # [128]
                        smooth_value = (1 / (smooth_label_nums + 1e-10) )*args.smooth_value

                        ## construct smooth label
                        expanded_targets = expanded_targets * smooth_value.unsqueeze(1)

                        ## 设置target_a 和 target_b 在smooth label中的标签
                        #import ipdb; ipdb.set_trace()
                        expanded_targets.scatter_(1, y_T, 1 - args.smooth_value)
                        
                        expanded_targets = expanded_targets.unsqueeze(1).expand(-1, action_preds.shape[1], -1).contiguous().view(-1, expanded_targets.shape[-1]) # [bs*len]

                        linear_action_preds = linear_action_preds.float()
                        log_action_preds = F.log_softmax(linear_action_preds, dim=1)                  
                        clc_loss = -torch.sum(log_action_preds * expanded_targets, dim=1)
                        clc_loss = clc_loss.mean()
                    
                    else:
                        ## anticipation: predicted_uncertainties [bs, len, C], others [bs]
                        action_preds, predicted_uncertainties, mixup_index = model(x, mode)

                        action_preds = action_preds[:, -args.S_ant:, :].contiguous()
                        predicted_uncertainties = predicted_uncertainties[:, -args.S_ant:, :].contiguous()

                        ## mixup label
                        targets_a = y
                        targets_b = y[mixup_index]

                        ## mixup uncertainty
                        predicted_uncertainties = predicted_uncertainties.mean(dim=2, keepdim=True)
                        combined_uncertainties = torch.min(torch.stack([predicted_uncertainties, predicted_uncertainties[mixup_index]], dim=0), dim=0)[0]
                        #scaled_action_preds = action_preds / F.sigmoid(combined_uncertainties)
                        scaled_action_preds = action_preds / combined_uncertainties
                        linear_action_preds = scaled_action_preds.view(-1, scaled_action_preds.shape[-1])
                        
                        ## expand smooth label
                        # for targets_a
                        statistic_action_label_a = batch['statistic_action_label'].to(device) # [bs, c]
                        statistic_action_label_a = (statistic_action_label_a > 0).int()
                        conceptnet_action_label_a = batch['conceptnet_action_label'].to(device)
                        conceptnet_action_label_a = (conceptnet_action_label_a > 0).int()
                        expanded_targets_a = statistic_action_label_a + conceptnet_action_label_a

                        # for targets_b
                        statistic_action_label_b = statistic_action_label_a[mixup_index]
                        statistic_action_label_b = (statistic_action_label_b > 0).int()
                        conceptnet_action_label_b = conceptnet_action_label_a[mixup_index]
                        conceptnet_action_label_b = (conceptnet_action_label_b > 0).int()
                        expanded_targets_b = statistic_action_label_b + conceptnet_action_label_b

                        # combine
                        expanded_targets = expanded_targets_a + expanded_targets_b # [128, 2513]
                        
                        ## 将target_a 和 target_b 在smooth label中置零
                        targets_a_T = targets_a.unsqueeze(0).T # 转置
                        expanded_targets.scatter_(1, targets_a_T, 0)
                        targets_b_T = targets_b.unsqueeze(0).T # 转置
                        expanded_targets.scatter_(1, targets_b_T, 0)
                        
                        # 计算smooth label的数量
                        smooth_label_nums = expanded_targets.sum(dim=1) # [128]
                        smooth_value = (1 / (smooth_label_nums + 1e-10) )*args.smooth_value

                        ## construct smooth label
                        expanded_targets = expanded_targets * smooth_value.unsqueeze(1)

                        ## 设置target_a 和 target_b 在smooth label中的标签
                        #import ipdb; ipdb.set_trace()
                        expanded_targets.scatter_(1, targets_a_T, (1 - args.smooth_value) / 2)
                        expanded_targets.scatter_(1, targets_b_T, (1 - args.smooth_value) / 2)

                        expanded_targets = expanded_targets.unsqueeze(1).expand(-1, action_preds.shape[1], -1).contiguous().view(-1, expanded_targets.shape[-1]) # [bs*len]

                        linear_action_preds = linear_action_preds.float()
                        log_action_preds = F.log_softmax(linear_action_preds, dim=1)                  
                        clc_loss = -torch.sum(log_action_preds * expanded_targets, dim=1)
                        clc_loss = clc_loss.mean()
                   
                    ## ranking loss
                    uncertainty_labels = torch.tensor(list(range(predicted_uncertainties.mean(dim=2).shape[1]))).view(1, -1).expand(bs, -1).to(device)
                    ranking_loss = listMLE(predicted_uncertainties.mean(dim=2), uncertainty_labels, eps=1e-10)
                
                    ## uncertainty wd loss
                    #wd_loss = (predicted_uncertainties ** 2).sum(dim=2).sum(dim=1).mean()
                    wd_loss = (predicted_uncertainties.mean(dim=2) ** 2).sum()
                    
                    ## total loss
                    loss = clc_loss + args.ranking_loss_weight * ranking_loss + args.wd_loss_weight * wd_loss
                    #print(clc_loss.cpu().data, ranking_loss.cpu().data, wd_loss.cpu().data)

                else:
                    action_preds = model(x)
                    
                    action_preds = action_preds[:, -args.S_ant:, :].contiguous()
                    linear_action_preds = action_preds.view(-1, action_preds.shape[-1])
                    linear_action_labels = y.view(-1, 1).expand(-1, action_preds.shape[1]).contiguous().view(-1)

                    clc_loss = ce_loss_function(linear_action_preds, linear_action_labels)
                    ranking_loss = 0
                    
                    ## total loss
                    loss = clc_loss

                # get the predictions for anticipation time = 1s (index -4) (anticipation)
                # or for the last time-step (100%) (early recognition)
                # top5 accuracy at 1s
                idx = -4 if args.task == 'anticipation' else -1
                # use top-5 for anticipation and top-1 for early recognition
                k = 5 if args.task == 'anticipation' else 1
                acc = topk_accuracy(action_preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0]*100

                # store the values in the meters to keep incremental averages
                loss_meter[mode].add(loss.item(), bs)
                if args.mt5r:
                    accuracy_meter[mode].add(action_preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy())
                else:
                    accuracy_meter[mode].add(acc, bs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # compute decimal epoch for logging
                e = epoch + i/len(train_loader)

                # log training during loop
                # avoid logging the very first batch. It can be biased.
                if mode == 'training' and i != 0 and i % args.display_every == 0:
                    log(mode, e, loss_meter[mode], accuracy_meter[mode])

            # log at the end of each epoch
            log(mode, epoch+1, loss_meter[mode], accuracy_meter[mode],
                max(accuracy_meter[mode].value(), best_perf) if mode == 'validation'
                else None, green=True)

        mode = 'validation'
        # enable gradients only if training
        with torch.set_grad_enabled(False):
            model.eval()

            for i, batch in enumerate(validation_loader):
                x = batch['past_features' if args.task ==
                            'anticipation' else 'action_features']

                if type(x) == list:
                    x = [xx.to(device) for xx in x]
                else:
                    x = x.to(device)

                y = batch['label'].to(device)

                bs = y.shape[0]  # batch size
                '''
                preds = model(x)
                preds = preds[:, -args.S_ant:, :].contiguous()
                '''
                #if (args.task == 'anticipation' and args.modality == 'fusion') or (args.sequence_completion):
                if args.sequence_completion:
                    preds = model(x)
                    preds = preds[:, -args.S_ant:, :].contiguous()
                else:
                    preds, predicted_uncertainties = model(x)

                    # take only last S_ant predictions
                    preds = preds[:, -args.S_ant:, :].contiguous()
                    predicted_uncertainties = predicted_uncertainties[:, -args.S_ant:, :].contiguous()
                    
                    predicted_uncertainties = predicted_uncertainties.mean(dim=2, keepdim=True)
                    #preds = preds / F.sigmoid(predicted_uncertainties)
                    preds = preds / predicted_uncertainties
                    
                # linearize predictions
                linear_preds = preds.view(-1, preds.shape[-1])
                # replicate the labels across timesteps and linearize
                linear_labels = y.view(-1, 1).expand(-1, preds.shape[1]).contiguous().view(-1)

                loss = F.cross_entropy(linear_preds, linear_labels)
                # get the predictions for anticipation time = 1s (index -4) (anticipation)
                # or for the last time-step (100%) (early recognition)
                # top5 accuracy at 1s
                idx = -4 if args.task == 'anticipation' else -1
                # use top-5 for anticipation and top-1 for early recognition
                k = 5 if args.task == 'anticipation' else 1
                acc = topk_accuracy(
                    preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0]*100

                # store the values in the meters to keep incremental averages
                loss_meter[mode].add(loss.item(), bs)
                if args.mt5r:
                    accuracy_meter[mode].add(preds[:, idx, :].detach().cpu().numpy(),
                                                y.detach().cpu().numpy())
                else:
                    accuracy_meter[mode].add(acc, bs)

            # log at the end of each epoch
            log(mode, epoch+1, loss_meter[mode], accuracy_meter[mode],
                max(accuracy_meter[mode].value(), best_perf) if mode == 'validation'
                else None, green=True)

        if accuracy_meter['training'].value() > break_flag:
            break_count += 1
        if break_count >=5:
            break

        if best_perf < accuracy_meter['validation'].value():
            best_perf = accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False

        # save checkpoint at the end of each train/val epoch
        save_model(model, epoch+1, accuracy_meter['validation'].value(), best_perf,
                   is_best=is_best)

def get_validation_ids():
    unseen_participants_ids = pd.read_csv(join(args.path_to_data, 'validation_unseen_participants_ids.csv'), names=['id'], squeeze=True)
    tail_verbs_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_verbs_ids.csv'), names=['id'], squeeze=True)
    tail_nouns_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_nouns_ids.csv'), names=['id'], squeeze=True)
    tail_actions_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_actions_ids.csv'), names=['id'], squeeze=True)

    return unseen_participants_ids, tail_verbs_ids, tail_nouns_ids, tail_actions_ids

def get_many_shot():
    """Get many shot verbs, nouns and actions for class-aware metrics (Mean Top-5 Recall)"""
    # read the list of many shot verbs
    many_shot_verbs = pd.read_csv(
        join(args.path_to_data, 'EPIC_many_shot_verbs.csv'))['verb_class'].values
    # read the list of many shot nouns
    many_shot_nouns = pd.read_csv(
        join(args.path_to_data, 'EPIC_many_shot_nouns.csv'))['noun_class'].values

    # read the list of actions
    actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
    # map actions to (verb, noun) pairs
    a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
               for a in actions.iterrows()}

    # create the list of many shot actions
    # an action is "many shot" if at least one
    # between the related verb and noun are many shot
    many_shot_actions = []
    for a, (v, n) in a_to_vn.items():
        if v in many_shot_verbs or n in many_shot_nouns:
            many_shot_actions.append(a)

    return many_shot_verbs, many_shot_nouns, many_shot_actions


def main():
    model = get_model()
    if type(model) == list:
        model = [m.to(device) for m in model]
    else:
        model.to(device)

    if args.mode == 'train':
        if args.resume:
            start_epoch, _, start_best_perf = load_checkpoint(model)
        else:
            start_epoch = 0
            start_best_perf = 0

        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)

        with open(os.path.join(args.save_path, 'args.json'), 'w') as args_file:
            json.dump(vars(args), args_file)
            
        trainval(model, optimizer, args.epochs,
                 start_epoch, start_best_perf)

    elif args.mode == 'validate' and args.mcdropout:
        validation_set = get_loader('validation')
        loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

        epoch, perf, _ = load_checkpoint(model, best=True)
        print(f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")
        
        verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels, ids = get_scores_mcdropout(model, loader, include_discarded=args.ek100)
        
        verb_accuracies = topk_accuracy_multiple_timesteps(
            verb_scores, verb_labels)
        noun_accuracies = topk_accuracy_multiple_timesteps(
            noun_scores, noun_labels)
        action_accuracies = topk_accuracy_multiple_timesteps(
            action_scores, action_labels)

        many_shot_verbs, many_shot_nouns, many_shot_actions = get_many_shot()

        verb_recalls = topk_recall_multiple_timesteps(
            verb_scores, verb_labels, k=5, classes=many_shot_verbs)
        noun_recalls = topk_recall_multiple_timesteps(
            noun_scores, noun_labels, k=5, classes=many_shot_nouns)
        action_recalls = topk_recall_multiple_timesteps(
            action_scores, action_labels, k=5, classes=many_shot_actions)

        all_accuracies = np.concatenate(
            [verb_accuracies, noun_accuracies, action_accuracies, verb_recalls, noun_recalls, action_recalls])
        all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
        indices = [
            ('Verb', 'Top-1 Accuracy'),
            ('Verb', 'Top-5 Accuracy'),
            ('Verb', 'Mean Top-5 Recall'),
            ('Noun', 'Top-1 Accuracy'),
            ('Noun', 'Top-5 Accuracy'),
            ('Noun', 'Mean Top-5 Recall'),
            ('Action', 'Top-1 Accuracy'),
            ('Action', 'Top-5 Accuracy'),
            ('Action', 'Mean Top-5 Recall'),
        ]

        if args.task == 'anticipation':
            cc = np.linspace(args.alpha*args.S_ant, args.alpha, args.S_ant, dtype=str)
        else:
            cc = [f"{c:0.1f}%" for c in np.linspace(0,100,args.S_ant+1)[1:]]

        scores = pd.DataFrame(all_accuracies*100, columns=cc, index=pd.MultiIndex.from_tuples(indices))

        print(scores)

        if args.task == 'anticipation':
            tta_verb = tta(verb_scores, verb_labels)
            tta_noun = tta(noun_scores, noun_labels)
            tta_action = tta(action_scores, action_labels)

            print(f"\nMean TtA(5): VERB: {tta_verb:0.2f} NOUN: {tta_noun:0.2f} ACTION: {tta_action:0.2f}")

    elif args.mode == 'validate':
        if args.modality == 'fusion' and args.simple_fusion:
            if args.egategazeplus:
                validation_sets = [get_loader('validation', 'rgb_s' + str(args.split_num)), get_loader('validation', 'flow_s' + str(args.split_num))]
            else:
                validation_sets = [get_loader('validation', 'rgb'), get_loader('validation', 'flow'), get_loader('validation', 'obj')]
            loaders =[torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False) for validation_set in validation_sets]

            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels, ids = get_scores_simple_fusion(model, loaders)
        else:
            validation_set = get_loader('validation')
            loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

            epoch, perf, _ = load_checkpoint(model, best=True)
            print(f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")
                
            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels, ids = get_scores(model, loader, include_discarded=args.ek100)
             
        if args.ek100:
            overall_verb_recalls = topk_recall_multiple_timesteps(
                verb_scores, verb_labels, k=5)
            overall_noun_recalls = topk_recall_multiple_timesteps(
                noun_scores, noun_labels, k=5)
            overall_action_recalls = topk_recall_multiple_timesteps(
                action_scores, action_labels, k=5)

            unseen, tail_verbs, tail_nouns, tail_actions = get_validation_ids()

            unseen_bool_idx = pd.Series(ids).isin(unseen).values
            tail_verbs_bool_idx = pd.Series(ids).isin(tail_verbs).values
            tail_nouns_bool_idx = pd.Series(ids).isin(tail_nouns).values
            tail_actions_bool_idx = pd.Series(ids).isin(tail_actions).values

            tail_verb_recalls = topk_recall_multiple_timesteps(
                verb_scores[tail_verbs_bool_idx], verb_labels[tail_verbs_bool_idx], k=5)
            tail_noun_recalls = topk_recall_multiple_timesteps(
                noun_scores[tail_nouns_bool_idx], noun_labels[tail_nouns_bool_idx], k=5)
            tail_action_recalls = topk_recall_multiple_timesteps(
                action_scores[tail_actions_bool_idx], action_labels[tail_actions_bool_idx], k=5)

            unseen_verb_recalls = topk_recall_multiple_timesteps(
                verb_scores[unseen_bool_idx], verb_labels[unseen_bool_idx], k=5)
            unseen_noun_recalls = topk_recall_multiple_timesteps(
                noun_scores[unseen_bool_idx], noun_labels[unseen_bool_idx], k=5)
            unseen_action_recalls = topk_recall_multiple_timesteps(
                action_scores[unseen_bool_idx], action_labels[unseen_bool_idx], k=5)

            all_accuracies = np.concatenate(
                [overall_verb_recalls, overall_noun_recalls, overall_action_recalls, unseen_verb_recalls, unseen_noun_recalls, unseen_action_recalls, tail_verb_recalls, tail_noun_recalls, tail_action_recalls]
            ) #9 x 8

            #all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
            indices = [
                ('Overall Mean Top-5 Recall', 'Verb'),
                ('Overall Mean Top-5 Recall', 'Noun'),
                ('Overall Mean Top-5 Recall', 'Action'),
                ('Unseen Mean Top-5 Recall', 'Verb'),
                ('Unseen Mean Top-5 Recall', 'Noun'),
                ('Unseen Mean Top-5 Recall', 'Action'),
                ('Tail Mean Top-5 Recall', 'Verb'),
                ('Tail Mean Top-5 Recall', 'Noun'),
                ('Tail Mean Top-5 Recall', 'Action'),
            ]

            if args.task == 'anticipation':
                cc = np.linspace(args.alpha*args.S_ant, args.alpha, args.S_ant, dtype=str)
            else:
                cc = [f"{c:0.1f}%" for c in np.linspace(0,100,args.S_ant+1)[1:]]

            scores = pd.DataFrame(all_accuracies*100, columns=cc, index=pd.MultiIndex.from_tuples(indices))

        else:
            if args.egategazeplus:
                verb_accuracies = topk_accuracy_multiple_timesteps(
                    verb_scores, verb_labels)
                noun_accuracies = topk_accuracy_multiple_timesteps(
                    noun_scores, noun_labels)
                action_accuracies = topk_accuracy_multiple_timesteps(
                    action_scores, action_labels)

                verb_recalls = topk_recall_multiple_timesteps(
                    verb_scores, verb_labels, k=5)
                noun_recalls = topk_recall_multiple_timesteps(
                    noun_scores, noun_labels, k=5)
                action_recalls = topk_recall_multiple_timesteps(
                    action_scores, action_labels, k=5)

                all_accuracies = np.concatenate(
                    [verb_accuracies, noun_accuracies, action_accuracies, verb_recalls, noun_recalls, action_recalls])
                all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
                indices = [
                    ('Verb', 'Top-1 Accuracy'),
                    ('Verb', 'Top-5 Accuracy'),
                    ('Verb', 'Mean Top-5 Recall'),
                    ('Noun', 'Top-1 Accuracy'),
                    ('Noun', 'Top-5 Accuracy'),
                    ('Noun', 'Mean Top-5 Recall'),
                    ('Action', 'Top-1 Accuracy'),
                    ('Action', 'Top-5 Accuracy'),
                    ('Action', 'Mean Top-5 Recall'),
                ]

                if args.task == 'anticipation':
                    cc = np.linspace(args.alpha*args.S_ant, args.alpha, args.S_ant, dtype=str)
                else:
                    cc = [f"{c:0.1f}%" for c in np.linspace(0,100,args.S_ant+1)[1:]]

                scores = pd.DataFrame(all_accuracies*100, columns=cc, index=pd.MultiIndex.from_tuples(indices))

            else:
                verb_accuracies = topk_accuracy_multiple_timesteps(
                    verb_scores, verb_labels)
                noun_accuracies = topk_accuracy_multiple_timesteps(
                    noun_scores, noun_labels)
                action_accuracies = topk_accuracy_multiple_timesteps(
                    action_scores, action_labels)

                many_shot_verbs, many_shot_nouns, many_shot_actions = get_many_shot()

                verb_recalls = topk_recall_multiple_timesteps(
                    verb_scores, verb_labels, k=5, classes=many_shot_verbs)
                noun_recalls = topk_recall_multiple_timesteps(
                    noun_scores, noun_labels, k=5, classes=many_shot_nouns)
                action_recalls = topk_recall_multiple_timesteps(
                    action_scores, action_labels, k=5, classes=many_shot_actions)

                all_accuracies = np.concatenate(
                    [verb_accuracies, noun_accuracies, action_accuracies, verb_recalls, noun_recalls, action_recalls])
                all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
                indices = [
                    ('Verb', 'Top-1 Accuracy'),
                    ('Verb', 'Top-5 Accuracy'),
                    ('Verb', 'Mean Top-5 Recall'),
                    ('Noun', 'Top-1 Accuracy'),
                    ('Noun', 'Top-5 Accuracy'),
                    ('Noun', 'Mean Top-5 Recall'),
                    ('Action', 'Top-1 Accuracy'),
                    ('Action', 'Top-5 Accuracy'),
                    ('Action', 'Mean Top-5 Recall'),
                ]

                if args.task == 'anticipation':
                    cc = np.linspace(args.alpha*args.S_ant, args.alpha, args.S_ant, dtype=str)
                else:
                    cc = [f"{c:0.1f}%" for c in np.linspace(0,100,args.S_ant+1)[1:]]

                scores = pd.DataFrame(all_accuracies*100, columns=cc, index=pd.MultiIndex.from_tuples(indices))

        print(scores)

        if args.task == 'anticipation':
            tta_verb = tta(verb_scores, verb_labels)
            tta_noun = tta(noun_scores, noun_labels)
            tta_action = tta(action_scores, action_labels)

            print(
                f"\nMean TtA(5): VERB: {tta_verb:0.2f} NOUN: {tta_noun:0.2f} ACTION: {tta_action:0.2f}")

    elif 'test' in args.mode:
        if args.ek100:
            mm = ['timestamps']
        else:
            mm = ['seen', 'unseen']
        for m in mm:
            if args.task == 'early_recognition' and args.modality == 'fusion':
                loaders = [get_loader(f"test_{m}", 'rgb'), get_loader(f"test_{m}", 'flow'), get_loader(f"test_{m}", 'obj')]
                discarded_ids = loaders[0].dataset.discarded_ids
                verb_scores, noun_scores, action_scores, ids = get_scores_early_recognition_fusion(model, loaders)
            else:
                test_set = get_loader(f"test_{m}")
                loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

                epoch, perf, _ = load_checkpoint(model, best=True)

                discarded_ids = loader.dataset.discarded_ids

                print(
                    f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")

                verb_scores, noun_scores, action_scores, ids = get_scores(model, loader)

            idx = -4 if args.task == 'anticipation' else -1
            #import ipdb; ipdb.set_trace()
            ids = list(ids) + list(discarded_ids)
            verb_scores = np.concatenate((verb_scores, np.zeros((len(discarded_ids), *verb_scores.shape[1:])))) [:,idx,:]
            noun_scores = np.concatenate((noun_scores, np.zeros((len(discarded_ids), *noun_scores.shape[1:])))) [:,idx,:]
            action_scores = np.concatenate((action_scores, np.zeros((len(discarded_ids), *action_scores.shape[1:])))) [:,idx,:]

            actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
            # map actions to (verb, noun) pairs
            a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
                       for a in actions.iterrows()}

            import ipdb; ipdb.set_trace()
            preds = predictions_to_json(verb_scores, noun_scores, action_scores, ids, a_to_vn, version = '0.2' if args.ek100 else '0.1', sls=True)
            #import ipdb; ipdb.set_trace()
            if args.ek100:
                with open(join(args.json_directory,exp_name+f"_test.json"), 'w') as f:
                    f.write(json.dumps(preds, indent=4, separators=(',',': ')))
            else:
                with open(join(args.json_directory,exp_name+f"_{m}.json"), 'w') as f:
                    f.write(json.dumps(preds, indent=4, separators=(',',': ')))
    
    elif 'validate_json' in args.mode:
        if args.task == 'early_recognition' and args.modality == 'fusion':
            loaders = [get_loader("validation", 'rgb'), get_loader("validation", 'flow'), get_loader("validation", 'obj')]
            discarded_ids = loaders[0].dataset.discarded_ids
            verb_scores, noun_scores, action_scores, ids = get_scores_early_recognition_fusion(model, loaders)
        else:
            validation_set = get_loader('validation')
            loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

            epoch, perf, _ = load_checkpoint(model, best=True)            
            discarded_ids = loader.dataset.discarded_ids

            print(
                f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")

            verb_scores, noun_scores, action_scores, ids = get_scores(model, loader, challenge=True)

        idx = -4 if args.task == 'anticipation' else -1
        ids = list(ids) + list(discarded_ids)
        verb_scores = np.concatenate((verb_scores, np.zeros((len(discarded_ids), *verb_scores.shape[1:])))) [:,idx,:]
        noun_scores = np.concatenate((noun_scores, np.zeros((len(discarded_ids), *noun_scores.shape[1:])))) [:,idx,:]
        action_scores = np.concatenate((action_scores, np.zeros((len(discarded_ids), *action_scores.shape[1:])))) [:,idx,:]

        actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
        # map actions to (verb, noun) pairs
        a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
                   for a in actions.iterrows()}

        preds = predictions_to_json(verb_scores, noun_scores, action_scores, ids, a_to_vn, version = '0.2' if args.ek100 else '0.1', sls=True)

        with open(join(args.json_directory,exp_name+f"_validation.json"), 'w') as f:
            f.write(json.dumps(preds, indent=4, separators=(',',': ')))

if __name__ == '__main__':
    main()
