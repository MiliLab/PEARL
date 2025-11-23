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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str
    # optional decoded probe response
    probe_response: Optional[str]
    # optional probe ground truth
    probe_ground_truth: Optional[str]


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []

        # 安全取值，如果键不存在则使用空 tensor
        response_ids = data.batch.get("responses", torch.tensor([]))
        response_mask = data.batch.get("response_mask", torch.tensor([]))

        # 计算响应长度；若 mask 为空则给一个全零长度
        if response_mask.numel() > 0:
            response_length = torch.sum(response_mask, dim=-1)
        else:
            response_length = torch.zeros(len(response_ids), dtype=torch.long)

        print("####################function.py:119####################", data.batch.keys())

        # 检查 probe 响应
        has_probe = "responses_probe" in data.batch
        probe_response_ids = data.batch.get("responses_probe", None)
        if has_probe and probe_response_ids is not None and probe_response_ids.dim() == 3:
            probe_response_ids = probe_response_ids[:, 0, :]

        # 迭代样本
        for i in range(len(data)):
            cur_response_length = (
                int(response_length[i].item()) if i < len(response_length) else 0
            )

            # 安全切片
            valid_response_ids = (
                response_ids[i][:cur_response_length]
                if i < len(response_ids) and cur_response_length > 0
                else []
            )

            # 安全 decode
            try:
                response_str = self.tokenizer.decode(
                    valid_response_ids,
                    skip_special_tokens=getattr(self.config, "skip_special_tokens", True),
                )
            except Exception:
                response_str = ""

            # ground truth
            ground_truth_list = data.non_tensor_batch.get("ground_truth", [])
            ground_truth = ground_truth_list[i] if i < len(ground_truth_list) else None

            reward_input: dict[str, Optional[str] | int] = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": ground_truth,
            }

            # probe 相关（可选）
            if has_probe and probe_response_ids is not None and i < len(probe_response_ids):
                probe_ids_i = probe_response_ids[i]
                try:
                    probe_str = self.tokenizer.decode(
                        probe_ids_i,
                        skip_special_tokens=getattr(self.config, "skip_special_tokens", True),
                    )
                except Exception:
                    probe_str = ""
                reward_input["response_probe"] = probe_str

                gt_probe_list = data.non_tensor_batch.get("ground_truth_probe", [])
                reward_input["ground_truth_probe"] = (
                    gt_probe_list[i] if i < len(gt_probe_list) else None
                )

            reward_inputs.append(reward_input)

        # 调用 reward_fn（如果没定义就返回空）
        scores = self.reward_fn(reward_inputs) if hasattr(self, "reward_fn") else []

        # 初始化 reward tensor（大小匹配 response_ids）
        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        reward_tensor_probe = (
            torch.zeros_like(probe_response_ids, dtype=torch.float32)
            if has_probe and probe_response_ids is not None
            else None
        )

        reward_metrics = defaultdict(list)

        # 处理打分结果
        for i, score in enumerate(scores or []):
            cur_response_length = (
                int(response_length[i].item()) if i < len(response_length) else 0
            )
            if cur_response_length > 0 and "overall" in score:
                reward_tensor[i, cur_response_length - 1] = score.get("overall", 0.0)
            if has_probe and reward_tensor_probe is not None:
                reward_tensor_probe[i, cur_response_length - 1] = score.get("probe_accuracy", 0.0)
            for key, value in (score or {}).items():
                reward_metrics[key].append(value)

        # 返回
        if has_probe and reward_tensor_probe is not None:
            return reward_tensor, reward_tensor_probe, reward_metrics
        else:
            return reward_tensor,None, reward_metrics



    # def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
    #     reward_inputs = []
    #     response_ids = data.batch["responses"]
    #     response_length = torch.sum(data.batch["response_mask"], dim=-1)
    #     print("####################function.py:119####################",data.batch.keys())
    #     # optional probe responses produced in trainer; look for `responses_probe`
    #     has_probe = "responses_probe" in data.batch
    #     if has_probe:
    #         probe_response_ids = data.batch["responses_probe"]
    #         # if probe generation is n=1, shape may be [B, L]; support both
    #         if probe_response_ids.dim() == 3:
    #             probe_response_ids = probe_response_ids[:, 0, :]
    #     for i in range(len(data)):
    #         cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
    #         valid_response_ids = response_ids[i][:cur_response_length]
    #         response_str = self.tokenizer.decode(
    #             valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
    #         )
    #         reward_input: dict[str, Optional[str] | int] = {
    #             "response": response_str,
    #             "response_length": cur_response_length,
    #             "ground_truth": data.non_tensor_batch["ground_truth"][i],
    #         }
    #         if has_probe:
    #             # decode whole probe response sequence for the same sample i
    #             probe_ids_i = probe_response_ids[i]
    #             probe_str = self.tokenizer.decode(
    #                 probe_ids_i, skip_special_tokens=self.config.skip_special_tokens
    #             )
    #             reward_input["response_probe"] = probe_str
    #             # attach probe ground truth when provided by dataset
    #             if "ground_truth_probe" in data.non_tensor_batch:
    #                 reward_input["ground_truth_probe"] = data.non_tensor_batch["ground_truth_probe"][i]
    #         reward_inputs.append(reward_input)  # type: ignore[arg-type]

    #     scores = self.reward_fn(reward_inputs)
    #     reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    #     if has_probe:
    #         reward_tensor_probe = torch.zeros_like(data.batch["responses_probe"], dtype=torch.float32)
    #     reward_metrics = defaultdict(list)
    #     for i, score in enumerate(scores):
    #         cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
    #         reward_tensor[i, cur_response_length - 1] = score["overall"]
    #         if has_probe:
    #             reward_tensor_probe[i, cur_response_length - 1] = score["probe_accuracy"]
    #         for key, value in score.items():
    #             reward_metrics[key].append(value)
    #     if has_probe:
    #         return reward_tensor, reward_tensor_probe, reward_metrics
    #     else:
    #         return reward_tensor, reward_metrics