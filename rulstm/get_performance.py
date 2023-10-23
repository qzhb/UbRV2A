import os
import torch
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--print_params', default=['lr'], nargs='+')
args = parser.parse_args()


if 'ek100' in args.print_params:
    root_path = './models/ek100'
    if 'rgb' in args.print_params:
        model_name = 'RULSTM-anticipation_0.25_6_8_rgb_mt5r_best.pth.tar'
        sc_model_name = 'RULSTM-anticipation_0.25_6_8_rgb_mt5r_sequence_completion_best.pth.tar'
    elif 'flow' in args.print_params:
        model_name = 'RULSTM-anticipation_0.25_6_8_flow_mt5r_best.pth.tar'
        sc_model_name = 'RULSTM-anticipation_0.25_6_8_flow_mt5r_sequence_completion_best.pth.tar'
    elif 'obj' in args.print_params:
        model_name = 'RULSTM-anticipation_0.25_6_8_obj_mt5r_best.pth.tar'
        sc_model_name = 'RULSTM-anticipation_0.25_6_8_obj_mt5r_sequence_completion_best.pth.tar'
    elif 'fusion' in args.print_params:
        model_name = 'RULSTM-anticipation_0.25_6_8_fusion_mt5r_best.pth.tar'
        sc_model_name = 'RULSTM-anticipation_0.25_6_8_obj_mt5r_sequence_completion_best.pth.tar'
else: 
    if 'ek55' in args.print_params:
        root_path = './models/ek55'
    else:
        root_path = './models/egategazeplus'
    if 'rgb' in args.print_params:
        model_name = 'RULSTM-anticipation_0.25_6_8_rgb_best.pth.tar'
        sc_model_name = 'RULSTM-anticipation_0.25_6_8_rgb_sequence_completion_best.pth.tar'
    elif 'flow' in args.print_params:
        model_name = 'RULSTM-anticipation_0.25_6_8_flow_best.pth.tar'
        sc_model_name = 'RULSTM-anticipation_0.25_6_8_flow_sequence_completion_best.pth.tar'
    elif 'obj' in args.print_params:
        model_name = 'RULSTM-anticipation_0.25_6_8_obj_best.pth.tar'
        sc_model_name = 'RULSTM-anticipation_0.25_6_8_obj_sequence_completion_best.pth.tar'
    elif 'fusion' in args.print_params:
        model_name = 'RULSTM-anticipation_0.25_6_8_fusion_best.pth.tar'
        sc_model_name = 'RULSTM-anticipation_0.25_6_8_obj_sequence_completion_best.pth.tar'


time_list = sorted([temp for temp in os.listdir(root_path)])


sc_print_params = ['lr', 'dropout', 'modality', 'split_num']
for time_name in time_list:
    result_path = os.path.join(root_path, time_name, sc_model_name)
    if os.path.exists(result_path):
        result_dict = torch.load(result_path)

        params_path = os.path.join(root_path, time_name, 'args.json')
        params = {}
        with open(params_path, 'r') as f_data:
            params_dict = json.load(f_data)
            for key, value in params_dict.items():
                if key in sc_print_params:
                    params[key] = value

        print(time_name, result_dict['best_perf'], params)


performance_dict = {}
for time_name in time_list:
    result_path = os.path.join(root_path, time_name, model_name)
    if os.path.exists(result_path):
        result_dict = torch.load(result_path)

        params_path = os.path.join(root_path, time_name, 'args.json')
        params = {}
        with open(params_path, 'r') as f_data:
            params_dict = json.load(f_data)
            for key, value in params_dict.items():
                if key in args.print_params:
                    params[key] = value
        performance_dict[time_name] = [result_dict['epoch'], result_dict['best_perf'], params]
        #performance_dict[time_name] = params


sorted_performance_dict = sorted(performance_dict.items(), key = lambda x:x[1][1])

for info in sorted_performance_dict:
    print(info)
    print('\n')
