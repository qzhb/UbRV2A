from typing import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from termcolor import colored
from .transformer import transformer
from .lstm import lstm
from .model_utils import ClassifierWithLoss, ActionClassifierWithLoss



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


def build_reasoner(config,length):
    if config.name == 'transformer':
        return transformer(config, length)
    elif config.name == 'lstm':
        return lstm(config)
    else:
        raise NotImplementedError(config.name)


class DCR(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        
        self.config = config
        self.loss = self.config.loss

        self.full_frame = self.config.past_frame + self.config.anticipation_frame + self.config.action_frame

        self.encoder = nn.Linear(config.feat_dim, config.reasoner.d_model)
        self.decoder = nn.Linear(config.reasoner.d_model, config.feat_dim)
        
        self.reasoner = build_reasoner(config.reasoner, self.config.past_frame + self.config.anticipation_frame + self.config.action_frame)

        self.num_verb = dataset.num_verb
        self.num_noun = dataset.num_noun
        self.num_action = dataset.num_action

        #### 这两个参数，是为了resubmit补充实验临时添加的
        self.mixup = False
        self.ranking_loss = False
        self.expand_label = True
        self.ce_loss_function = torch.nn.CrossEntropyLoss()

        if config.classifier.verb:
            self.verb_classifier = ClassifierWithLoss(
                config.feat_dim,
                self.num_verb,
                hiddens = config.classifier.hidden,
                dropout = config.classifier.dropout,
                loss_smooth = config.loss.smooth,
                loss_weight = dataset.verb_weight,
            )
        else:
            self.verb_classifier = None
        
        if config.classifier.noun:
            self.noun_classifier = ClassifierWithLoss(
                config.feat_dim,
                self.num_noun,
                hiddens = config.classifier.hidden,
                dropout = config.classifier.dropout,
                loss_smooth = config.loss.smooth,
                loss_weight = dataset.noun_weight,
            )
        else:
            self.noun_classifier = None
        
        if config.classifier.action:
            self.action_classifier = ActionClassifierWithLoss(
            #self.action_classifier = ClassifierWithLoss(
                config.feat_dim,
                self.num_action,
                hiddens = config.classifier.hidden,
                dropout = config.classifier.dropout,
                loss_smooth = config.loss.smooth,
                loss_weight = dataset.action_weight,
                mixup = self.mixup,
            )
        else:
            self.action_classifier = None
    
    def generate_mask(self,easiness):
        mask = torch.ones(len(easiness),self.full_frame).to(easiness.device)
        mask[ : ,  -self.config.action_frame : ] = 0
        mask[ : ,  -(self.config.anticipation_frame + self.config.action_frame) : -self.config.action_frame] = \
            torch.bernoulli(easiness.unsqueeze(1).tile(self.config.anticipation_frame))
        return mask
    
        
    def forward(self,
        batch,
        is_training=False):
        
        frames = batch['past_frame']
        bz, cur_frame, dim = frames.shape
        
        assert cur_frame in [self.config.past_frame, self.full_frame]

        if cur_frame == self.config.past_frame:
            frames = torch.nn.functional.pad(frames, (0,0,0,self.config.anticipation_frame + self.config.action_frame))

        if not is_training:
            mask = torch.ones(self.full_frame).to(frames.device)
            mask[-(self.config.anticipation_frame + self.config.action_frame):] = 0
            out_frames = self.decoder(self.reasoner(self.encoder(frames), mask))

            consensus = OrderedDict()
            for i in range(1,1 + self.config.action_frame):
                pred = OrderedDict()
                if self.verb_classifier is not None:
                    pred['verb'] = self.verb_classifier(out_frames[:, - i])
                if self.noun_classifier is not None:
                    pred['noun'] = self.noun_classifier(out_frames[:, - i])
                if self.action_classifier is not None:
                    pred['action'] = self.action_classifier(out_frames[:, - i])

                #import ipdb; ipdb.set_trace()
                for k,v in pred.items():
                    if k not in consensus:
                        consensus[k] = v.clone()
                    else:
                        consensus[k] = consensus[k] + v
            
            #print('ok')
            return consensus
        else:
            # import ipdb; ipdb.set_trace()
            mask = self.generate_mask(batch['easiness']) # 128 * 48  最后四个要预测的为0
            out_frames = self.decoder(self.reasoner(self.encoder(frames), mask))
            
            loss_dict = OrderedDict()
            loss_dict['loss_total'] = 0
            if self.loss.next_cls > 0:
                if self.verb_classifier is not None:
                    loss_dict['loss_next_verb'] = 0
                    for i in range(1,1+self.config.action_frame):
                        loss_dict['loss_next_verb'] += self.verb_classifier(
                            out_frames[:, - i],
                            batch['next_verb_class']
                            ) * self.loss.verb# * self.loss.next_cls
                    loss_dict['loss_total'] += loss_dict['loss_next_verb']

                if self.noun_classifier is not None:
                    loss_dict['loss_next_noun'] = 0
                    for i in range(1,1+self.config.action_frame):
                        loss_dict['loss_next_noun'] += self.noun_classifier(
                            out_frames[:, - i],
                            batch['next_noun_class']
                            ) * self.loss.noun# * self.loss.next_cls
                    loss_dict['loss_total'] += loss_dict['loss_next_noun']

                '''
                if self.action_classifier is not None:
                    loss_dict['loss_next_action'] = 0
                    for i in range(1,1+self.config.action_frame):
                        loss_dict['loss_next_action'] += self.action_classifier(
                            out_frames[:, - i],
                            batch['next_action_class']
                            ) * self.loss.action * self.loss.next_cls
                    loss_dict['loss_total'] += loss_dict['loss_next_action']
                '''

                if self.action_classifier is not None:
                    
                    loss_dict['loss_next_action'] = 0
                    uncertainties = []
                    classifier_loss = 0
                    uncertainty_wd_loss = 0

                    for i in range(1, 1+self.config.action_frame):
                        
                        if self.mixup:
                            action_preds, y, predicted_uncertainties, mixup_index = self.action_classifier(out_frames[:, - i], batch['next_action_class'])

                            uncertainties.append(predicted_uncertainties)
                        
                            ## mixup label
                            targets_a = y
                            targets_b = y[mixup_index]

                            ## mixup uncertainty
                            predicted_uncertainties = predicted_uncertainties.mean(dim=1, keepdim=True)
                            
                            #predicted_uncertainties = F.softmax(predicted_uncertainties, dim=1)
                            #predicted_uncertainties = -torch.sum(predicted_uncertainties * torch.log2(predicted_uncertainties + 1e-10), dim=1)
                            #predicted_uncertainties = predicted_uncertainties.unsqueeze(1)

                            #combined_uncertainties = torch.min(torch.stack([predicted_uncertainties, predicted_uncertainties[mixup_index]], dim=0), dim=0)[0]
                            #combined_uncertainties = torch.max(torch.stack([predicted_uncertainties, predicted_uncertainties[mixup_index]], dim=0), dim=0)[0]
                            combined_uncertainties = predicted_uncertainties + predicted_uncertainties[mixup_index]
                            #combined_uncertainties = (predicted_uncertainties + predicted_uncertainties[mixup_index]) / 2
                            
                            linear_action_preds = action_preds / combined_uncertainties
                            
                            ## expand smooth label
                            # for targets_a
                            statistic_action_label_a = batch['statistic_action_label'].to('cuda') # [bsz, c]
                            statistic_action_label_a = (statistic_action_label_a > 0).int()
                            conceptnet_action_label_a = batch['conceptnet_action_label'].to('cuda')
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
                            smooth_value = (1 / (smooth_label_nums + 1e-10) )*self.loss.smooth_value

                            ## construct smooth label
                            expanded_targets = expanded_targets * smooth_value.unsqueeze(1)

                            ## 设置target_a 和 target_b 在smooth label中的标签
                            expanded_targets.scatter_(1, targets_a_T, (1 - self.loss.smooth_value) / 2)
                            expanded_targets.scatter_(1, targets_b_T, (1 - self.loss.smooth_value) / 2)
                            
                            linear_action_preds = linear_action_preds.float()
                            log_action_preds = F.log_softmax(linear_action_preds, dim=1)                  
                            clc_loss = -torch.sum(log_action_preds * expanded_targets, dim=1)
                            clc_loss = clc_loss.mean()
                            classifier_loss += clc_loss
                        
                        else:
                            action_preds, y, predicted_uncertainties = self.action_classifier(out_frames[:, - i], batch['next_action_class'])

                            uncertainties.append(predicted_uncertainties)

                            ## mixup label
                            targets = y
                            
                            ## clc loss
                            predicted_uncertainties = predicted_uncertainties.mean(dim=1, keepdim=True)
                            linear_action_preds = action_preds / predicted_uncertainties
                            
                            if self.expand_label:
                                ## expand smooth label
                                # for targets_a
                                statistic_action_label_a = batch['statistic_action_label'].to('cuda') # [bsz, c]
                                statistic_action_label_a = (statistic_action_label_a > 0).int()
                                conceptnet_action_label_a = batch['conceptnet_action_label'].to('cuda')
                                conceptnet_action_label_a = (conceptnet_action_label_a > 0).int()
                                expanded_targets = statistic_action_label_a + conceptnet_action_label_a

                                ## 将target_a 和 target_b 在smooth label中置零
                                targets_a_T = targets.unsqueeze(0).T # 转置
                                expanded_targets.scatter_(1, targets_a_T, 0)
                                
                                # 计算smooth label的数量
                                smooth_label_nums = expanded_targets.sum(dim=1) # [128]
                                smooth_value = (1 / (smooth_label_nums + 1e-10) )*self.loss.smooth_value

                                ## construct smooth label
                                expanded_targets = expanded_targets * smooth_value.unsqueeze(1)

                                ## 设置target_a 和 target_b 在smooth label中的标签
                                #import ipdb; ipdb.set_trace()
                                expanded_targets.scatter_(1, targets_a_T, 1 - self.loss.smooth_value)
                                linear_action_preds = linear_action_preds.float()
                                log_action_preds = F.log_softmax(linear_action_preds, dim=1)                  
                                clc_loss = -torch.sum(log_action_preds * expanded_targets, dim=1)
                                clc_loss = clc_loss.mean()
                                classifier_loss += clc_loss
                            else:
                                clc_loss = self.ce_loss_function(linear_action_preds, targets)
                                classifier_loss += clc_loss

                        ## uncertainty wd loss
                        wd_loss = (predicted_uncertainties.mean(dim=1) ** 2).sum()
                        uncertainty_wd_loss += wd_loss
                    
                    loss_dict['loss_next_action'] = classifier_loss * self.loss.action * self.loss.next_cls
                    loss_dict['loss_wd_action'] = self.loss.wd_loss_weight * uncertainty_wd_loss# * self.loss.next_cls
                    
                    if self.ranking_loss:
                        ## ranking loss
                        uncertainties = torch.stack(uncertainties, dim=1)  # [bsz, len, dim]
                        uncertainty_labels = torch.tensor(list(range(uncertainties.mean(dim=2).shape[1]))).view(1, -1).expand(uncertainties.shape[0], -1).to('cuda')
                        ranking_loss = listMLE(uncertainties.mean(dim=2), uncertainty_labels, eps=1e-10)
                        loss_dict['loss_ranking_action'] = self.loss.ranking_loss_weight * ranking_loss# * self.loss.next_cls
                        
                        ## total loss
                        loss_dict['loss_total'] += (loss_dict['loss_next_action'] + loss_dict['loss_ranking_action'] + loss_dict['loss_wd_action'])
                    else:
                        loss_dict['loss_total'] += (loss_dict['loss_next_action'] + loss_dict['loss_wd_action'])

            if self.loss.feat_mse > 0 :
                mse = nn.functional.mse_loss(
                    frames,
                    out_frames,
                    reduction='none'
                ).mean(-1) * (1 - mask)
                loss_dict['loss_feat_mse'] = mse.mean(0).sum() * self.loss.feat_mse
                loss_dict['loss_total'] += loss_dict['loss_feat_mse']
            loss_dict['MaskedFrame'] = (1 - mask).float().sum() / bz
            
            last_visible = torch.arange(self.full_frame).expand(bz,self.full_frame).to(frames.device) * mask
            last_visible = last_visible.max(-1)[0]
            assert torch.any(self.full_frame - last_visible > self.config.action_frame)
            criterion_forward = 4
            criterion_index = (last_visible + criterion_forward).long()
            criterion = nn.functional.mse_loss(
                frames.detach()[torch.arange(bz).to(frames.device).long(),       criterion_index],
                out_frames.detach()[torch.arange(bz).to(frames.device).long(),   criterion_index],
                reduction='none'
            ).mean(-1)
            
            return loss_dict, criterion
