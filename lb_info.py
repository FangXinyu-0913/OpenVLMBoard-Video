import json
import pandas as pd
from collections import defaultdict
import gradio as gr
import copy as cp
import numpy as np

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

# CONSTANTS-URL
RESULT_FILE = '../video_leaderboard_result_final.json'
URL = "http://opencompass.openxlab.space/utils/OpenVLM.json"
VLMEVALKIT_README = 'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/README.md'
# CONSTANTS-CITATION
CITATION_BUTTON_TEXT = r"""@misc{duan2024vlmevalkitopensourcetoolkitevaluating,
      title={VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models}, 
      author={Haodong Duan and Junming Yang and Yuxuan Qiao and Xinyu Fang and Lin Chen and Yuan Liu and Amit Agarwal and Zhe Chen and Mo Li and Yubo Ma and Hailong Sun and Xiangyu Zhao and Junbo Cui and Xiaoyi Dong and Yuhang Zang and Pan Zhang and Jiaqi Wang and Dahua Lin and Kai Chen},
      year={2024},
      eprint={2407.11691},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11691}, 
}"""
CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
# CONSTANTS-TEXT
LEADERBORAD_INTRODUCTION = """# OpenVLM Video Leaderboard
### Welcome to the OpenVLM Video Leaderboard! On this leaderboard we share the evaluation results of VLMs on the video understanding benchmark obtained by the OpenSource Framework [**VLMEvalKit**](https://github.com/open-compass/VLMEvalKit) 🏆 
### Currently, OpenVLM Video Leaderboard covers {} different VLMs (including GPT-4o, Gemini-1.5, LLaVA-OneVision, etc.) and {} different video understanding benchmarks. 

This leaderboard was last updated: {}. 
"""
# CONSTANTS-FIELDS
META_FIELDS = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model', 'OpenSource', 'Verified', 'Frames']
MAIN_FIELDS = ['MVBench', 'Video-MME (w/o subs)', 'MMBench-Video']
MODEL_SIZE = ['<10B', '10B-20B', '20B-40B', '>40B', 'Unknown']
MODEL_TYPE = ['API', 'OpenSource']

# The README file for each benchmark
LEADERBOARD_MD = {}

LEADERBOARD_MD['MAIN'] = """
## Main Evaluation Results

- Avg Score: The average score on all video understanding Benchmarks (normalized to 0 - 100, the higher the better). 
- Avg Rank: The average rank on all video understanding Benchmarks (the lower the better). 
- The overall evaluation results on 3 video understanding benchmarks, sorted by the ascending order of Avg Rank. 
"""

LEADERBOARD_MD['Video-MME (w/o subs)'] = """
## Video-MME (w/o subs) Evaluation Results

- We give the total scores for the three video lengths (short, medium and long), as well as the total scores for each task type.
- Video-MME (w subs) will update as evaluation is completed.
"""

# LEADERBOARD_MD['MVBench'] = """
# ## MVBench Evaluation Results
# """

# LEADERBOARD_MD['MMBench-Video'] = """
# ## MMBench-Video Evaluation Results
# """



from urllib.request import urlopen

def load_results():
    with open(RESULT_FILE, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def nth_large(val, vals):
    return sum([1 for v in vals if v > val]) + 1

def format_timestamp(timestamp):
    return timestamp[:2] + '.' + timestamp[2:4] + '.' + timestamp[4:6] + ' ' + timestamp[6:8] + ':' + timestamp[8:10] + ':' + timestamp[10:12]

def model_size_flag(sz, FIELDS):
    if pd.isna(sz) or sz == 'N/A':
        if 'Unknown' in FIELDS:
            return True
        else:
            return False
    sz = float(sz.replace('B','').replace('(LLM)',''))
    if '<10B' in FIELDS and sz < 10:
        return True
    if '10B-20B' in FIELDS and sz >= 10 and sz < 20:
        return True
    if '20B-40B' in FIELDS and sz >= 20 and sz < 40:
        return True
    if '>40B' in FIELDS and sz >= 40:
        return True
    return False

def model_type_flag(line, FIELDS):
    if 'OpenSource' in FIELDS and line['OpenSource'] == 'Yes':
        return True
    if 'API' in FIELDS and line['OpenSource'] == 'No':
        return True
    return False

def BUILD_L1_DF(results, fields):
    res = defaultdict(list)
    for i, m in enumerate(results):
        item = results[m]
        meta = item['META']
        for k in META_FIELDS:
            if k == 'Parameters (B)':
                param = meta['Parameters']
                res[k].append(param.replace('B', '') if param != '' else None)
                # res[k].append(float(param.replace('B', '')) if param != '' else None)
            elif k == 'Method':
                name, url = meta['Method']
                res[k].append(f'<a href="{url}">{name}</a>')
            else:
                res[k].append(meta[k])
        scores, ranks = [], []
        for d in fields:
            res[d].append(item[d]['Overall'])
            # scores.append(item[d]['Overall'])
            if d == 'MMBench-Video':
                scores.append(item[d]['Overall'] / 3 * 100)
            else:
                scores.append(item[d]['Overall'])
            ranks.append(nth_large(item[d]['Overall'], [x[d]['Overall'] for x in results.values()]))
        res['Avg Score'].append(round(np.mean(scores), 1))
        res['Avg Rank'].append(round(np.mean(ranks), 2))

    df = pd.DataFrame(res)
    df = df.sort_values('Avg Rank')
    
    check_box = {}
    check_box['essential'] = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model', 'Frames']
    check_box['required'] = ['Avg Score', 'Avg Rank']
    check_box['all'] = check_box['required'] + ['OpenSource', 'Verified'] + fields
    type_map = defaultdict(lambda: 'number')
    type_map['Method'] = 'html'
    type_map['Language Model'] = type_map['Vision Model'] = type_map['OpenSource'] = type_map['Verified'] = type_map['Frames'] = 'str'
    check_box['type_map'] = type_map
    return df, check_box
        
def BUILD_L2_DF(results, dataset):
    res = defaultdict(list)
    fields = list(list(results.values())[0][dataset].keys())
    non_overall_fields = [x for x in fields if 'Overall' not in x]
    overall_fields = [x for x in fields if 'Overall' in x]
    
    for m in results:
        item = results[m]
        meta = item['META']
        for k in META_FIELDS:
            if k == 'Parameters (B)':
                param = meta['Parameters']
                res[k].append(param.replace('B', '') if param != '' else None)
                # res[k].append(float(param.replace('B', '')) if param != '' else None)
            elif k == 'Method':
                name, url = meta['Method']
                res[k].append(f'<a href="{url}">{name}</a>')
            else:
                res[k].append(meta[k])
        fields = [x for x in fields]
    
        for d in non_overall_fields:
            res[d].append(item[dataset][d])
        for d in overall_fields:
            res[d].append(item[dataset][d])

    df = pd.DataFrame(res)
    df = df.sort_values('Overall')
    df = df.iloc[::-1]
    
    check_box = {}
    check_box['essential'] = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model', 'Frames']
    if dataset == 'MMBench-Video':
        check_box['required'] = overall_fields + ['Perception', 'Reasoning']
    elif 'Video-MME' in dataset:
        check_box['required'] = overall_fields + ['short', 'medium', 'long']
    else:
        check_box['required'] = overall_fields
    check_box['all'] = non_overall_fields + overall_fields
    type_map = defaultdict(lambda: 'number')
    type_map['Method'] = 'html'
    type_map['Language Model'] = type_map['Vision Model'] = type_map['OpenSource'] = type_map['Verified'] = type_map['Frames'] ='str'
    check_box['type_map'] = type_map
    return df, check_box