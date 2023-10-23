from collections import defaultdict
import json
from pandas.core import frame
import torch
import pandas as pd
import os
import pickle as pkl
import numpy as np
import cv2
import h5py
import tqdm
import lmdb
from functools import lru_cache


class MECCANO_DATASET(torch.utils.data.Dataset):
    def __init__(self, logger, config):
        super().__init__()
        
        self.data_root = './data/MECCANO'
        
        self.name  = config.name
        self.split = config.split
        self.config = config
        self.model_fps = config.fps 
        self.tau_a = config.tau_a
        
        self.feature = config.feature
        self.feature_fps = config.feature_fps
        self.feature_dim = config.feature_dim
        
        if config.name == 'MECCANO':
    
            if config.split == 'train':
                self.action_info = pd.read_csv(os.path.join(self.data_root,'training.csv'), header=None, names=['video','action','name_action', 'start','end'])

            elif config.split == 'valid':
                self.action_info = pd.read_csv(os.path.join(self.data_root,'validation.csv'), header=None, names=['video','action','name_action', 'start','end'])

            elif config.split == 'test':
                self.action_info = pd.read_csv(os.path.join(self.data_root,'test.csv'), header=None, names=['video','action','name_action', 'start','end'])

            else:
                raise NotImplementedError('Unknow split [%s] for dataset [%s]' % (config.split, config.name))

            with open(os.path.join('/data1/qzhb/datasets/MECCANO/MECCANO_action_statistic.pkl'), 'rb') as f:
                self.statistic_action_sim = pkl.load(f)

            with open(os.path.join('/data1/qzhb/datasets/MECCANO/MECCANO_action_conceptnet.pkl'), 'rb') as f:
                self.conceptnet_action_sim = pkl.load(f)

            self.action2vn = {}
            with open(os.path.join(self.data_root,'activity_verb_noun_index.txt'), 'r') as f:
                label_infos = f.readlines()
                for label_info in label_infos:
                    label_ids = label_info.strip().split(' ')
                    self.action2vn[int(label_ids[0])] = [int(label_ids[1]), int(label_ids[2])]
            
            self.num_action = len(self.action2vn)
            self.num_verb = len(set([value[0] for key, value in self.action2vn.items()]))
            self.num_noun = len(set([value[1] for key, value in self.action2vn.items()]))
                    
        else:
            raise NotImplementedError('Unknow dataset: %s' % config.name)
        
        self.verb_weight, self.noun_weight, self.action_weight = None, None, None

        ##### store source frame index
        assert config.past_frame >= 0
        self.data = []
        self.frame_label = defaultdict(dict)
        
        for video_id, group in self.action_info.groupby('video'):
            for idx, a in group.iterrows():
                segment = {
                    'id' : idx,
                    'video_id': video_id,
                }
                
                start_frame = int(a.start.strip().split('.jpg')[0])
                end_frame = int(a.end.strip().split('.jpg')[0])
                verb_class = self.action2vn[a.action][0]
                noun_class = self.action2vn[a.action][1]
                
                for fid in range(start_frame,end_frame):
                    self.frame_label[video_id][fid] = (verb_class, noun_class)

                if config.drop and start_frame<=self.tau_a * self.feature_fps:
                    continue
                
                frame_index = np.arange(
                    start_frame - self.tau_a * self.feature_fps + config.forward_frame * self.feature_fps / self.model_fps, 
                    start_frame - self.tau_a * self.feature_fps - config.past_frame * self.feature_fps / self.model_fps,
                    - self.feature_fps / self.model_fps
                ).astype(int)[::-1]
                
                assert len(frame_index) == config.past_frame + config.forward_frame
                frame_index[frame_index < 1] = 1
                
                segment['frame_index'] = frame_index

                segment['next_verb_class'] = verb_class
                segment['next_noun_class'] = noun_class
                segment['next_action_class'] = a.action

                self.data.append(segment)

        ##### feature
        assert config.feat_file
        self.f = lmdb.open(config.feat_file, readonly=True, lock=False)

        logger.info('[%s] # Frame: Past %d. Forward %d.' % (
            config.split, config.past_frame,config.forward_frame))
        logger.info('[%s] # segment %d. verb %d. noun %d. action %d.' % (
            config.split, len(self.data), self.num_verb, self.num_noun, self.num_action))

        self.cache = {}
        if config.cache:
            self.make_cache(logger)


    def make_cache(self,logger):
        logger.info('Cache: Load all feature into memory')
        for segment in self.data:
            for fid in segment['frame_index']:            
                key = '%s_frame_%010d.jpg' % (segment['video_id'],fid)
                if key not in self.cache:
                    res = self._read_one_frame_feat(key)
                    self.cache[key] = res
        logger.info('Cache: Finish loading. Cache Size %d' % len(self.cache))

    def timestr_to_second(self,x):
        a,b,c = list(map(float,x.split(':')))
        return c + 60 * b + 3600 * a

   
    def _read_one_frame_feat(self, key):

        ## reset key name for meccano dataset
        splited_key = key.split('_frame_')
        new_num = str(int(splited_key[1].split('.jpg')[0])).zfill(5)
        key = splited_key[0].zfill(2) + '_' + new_num + '.jpg'

        if key in self.cache:
            return self.cache[key]
        
        with self.f.begin() as e:
            buf = e.get(key.strip().encode('utf-8'))
            #if buf is not None:
            #    res = np.frombuffer(buf,'float32')
            #else:
            #    res = None

        while not buf:
            splited_key = key.split('_')
            new_num = str(int(splited_key[1].split('.jpg')[0])-1).zfill(5)
            new_key = splited_key[0].zfill(2) + '_' + new_num + '.jpg'
            with self.f.begin() as e:
                buf = e.get(new_key.strip().encode('utf-8'))
            key = new_key

        res = np.frombuffer(buf,'float32')
        
        return res
      
    def _load_feat(self,video_id, frame_ids):
        frames = []
        dim = self.feature_dim
        
        for fid in frame_ids:
            # handling special case for irCSN feature provided by AVT
            if self.feature == 'irCSN10':
                if fid %3!=0:
                    fid = (fid // 3) * 3 
            if self.feature == 'irCSN25':
                if fid % 6 == 3:
                    fid = fid -1
            
            key = '%s_frame_%010d.jpg' % (video_id,fid)
            frame_feat = self._read_one_frame_feat(key)
            if frame_feat is not None:
                frames.append(frame_feat)
            elif len(frames) > 0:
                frames.append(frames[-1])
                # print('Copy frame:    %s' % key)
            else:
                frames.append(np.zeros(dim))
                # print('Zero frame:    %s' % key)
        return torch.from_numpy(np.stack(frames,0)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        segment = self.data[i]

        out = {
            'id' : segment['id'],
            'index' : i
        }
        
        out['next_action_class'] = segment['next_action_class']
        out['next_verb_class'] = segment['next_verb_class']
        out['next_noun_class'] = segment['next_noun_class']
        
        label_sim = torch.tensor(list(self.statistic_action_sim[segment['next_action_class']].values()))
        out['statistic_action_label'] = label_sim.float()
        label_sim = torch.tensor(list(self.conceptnet_action_sim[str(segment['next_action_class'])].values()))
        out['conceptnet_action_label'] = label_sim.float()
        
        out['past_frame'] = self._load_feat(
            segment['video_id'], 
            segment['frame_index'], 
        )
       
        out['frame_index'] = list(segment['frame_index'])

        return out
