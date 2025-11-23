# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

def extract_answers_mqa(text):
    answer_pattern = r"<a(\d+)>(.*?)</a\1>"
    answers = re.findall(answer_pattern, text)
    
    return [answer[1] for answer in answers]

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def accuracy_reward_mqa(response: str, ground_truth: str) -> float:
    if '<boxed>' not in response:
        extract_resp_list = extract_answers_mqa(response)
        extract_gt_list = extract_answers_mqa(ground_truth)
    else:
        extract_resp_list = extract_boxed_content(response)
        extract_gt_list = extract_boxed_content(ground_truth)
    if len(extract_resp_list) == 0 or len(extract_gt_list) == 0:
        return 0.0
    else:
        correct_count = 0
        for gt, resp in zip(extract_gt_list, extract_resp_list):
            if grade_answer(gt, resp):
                correct_count += 1
        return correct_count / len(extract_resp_list)

def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    probe_weight: float = 0.0,
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        # original
        og_resp = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        og_format = format_reward(og_resp)
        og_acc = accuracy_reward(og_resp, reward_input["ground_truth"])
        # probe (optional)
        probe_resp = reward_input.get("response_probe")
        
        if probe_resp is not None:
            pr = re.sub(r"\s*(<|>|/)\s*", r"\1", probe_resp)
            probe_format = format_reward(pr)
            if "ground_truth_probe" in reward_input:
                probe_gt = reward_input["ground_truth_probe"]
                probe_acc = accuracy_reward_mqa(pr, probe_gt)
            else:
                probe_gt = reward_input["ground_truth"]
                probe_acc = accuracy_reward(pr, probe_gt)
            
            overall = (1 - format_weight) * og_acc + format_weight * og_format
            
            if probe_acc == 0 and og_acc == 1:
                lucky_rate = 1.0
            else:
                lucky_rate = 0.0
            if probe_acc == 1 and og_acc == 1:
                consistent_rate = 1.0
            else:
                consistent_rate = 0.0
            if probe_acc == 1:
                all_right = 1.0
            else:
                all_right=0.0
        else:
            probe_format = 0.0
            probe_acc = 0.0
            overall = (1 - format_weight) * og_acc + format_weight * og_format
            consistent_rate = 0.0
            lucky_rate = 0.0
            all_right = 0.0
        
        scores.append(
            {
                "overall": overall,
                "og_accuracy": og_acc,
                "og_format": og_format,
                "probe_accuracy": probe_acc,
                "probe_format": probe_format,
                "consistent_rate": consistent_rate,
                "lucky_rate": lucky_rate,
                "probe_all_right": all_right
            }
        )

    return scores
