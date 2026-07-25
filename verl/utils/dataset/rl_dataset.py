# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.import_utils import load_extern_object
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict: # J：自定义的 collate_fn 函数，用于将 batch 个的样本字典转换为同 key 名的一个整体 batch 的 Tensor 和 numpy 数组
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor): # J：如果是 Tensor 类型, 则聚合为一个列表，用于后续 stack
                tensors[key].append(val) # J：按照 key 聚合为一个列表，用于后续 stack
            else: # J：如果不是 Tensor 类型, 则聚合为一个列表，用于后续转换为 numpy 数组
                non_tensors[key].append(val) # J：按照 key 聚合为一个列表，用于后续转换为 numpy 数组

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0) # J：将聚合为一个列表的 Tensor 转换为一个 Tensor，形状为 (batch_size, *dims)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val)) # J：将聚合为一个列表的非 Tensor 类型值转换为 numpy 数组，形状为 (batch_size,)

    return {**tensors, **non_tensors} # J：返回一个字典，展开 tensors 和 non_tensors，暴露所有 key，包含所有聚合为一个列表的 Tensor 和非 Tensor 类型值


class RLHFDataset(Dataset): # J：默认使用的 RLHF 数据集类
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig, # J：这里得到的是根配置的 config.data 配置
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files) # J：深拷贝数据文件列表，用于后续操作
        self.original_data_files = copy.deepcopy(data_files)  # use for resume # J：深拷贝原始数据文件列表，用于恢复检查点
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        # J: 从配置中获取参数并赋值给实例变量
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.audio_key = config.get("audio_key", "audios")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {}) # J：apply_chat_template 的参数
        self.mm_processor_kwargs = config.get("mm_processor_kwargs", {})

        # Mirror AgentLoopWorker's tool loading so length filtering sees the
        # same schemas the rollout will.
        self.tool_config_path = config.get("tool_config_path", None)
        self.function_tool_path = config.get("function_tool_path", None)
        self.tool_schemas = None
        if self.tool_config_path or self.function_tool_path: # J：如果有工具配置路径或函数工具路径才初始化工具 schema
            try:
                from verl.tools.tool_registry import load_all_tools

                # J：加载预定义的工具，包含 Native Tools（比如沙盒执行工具）和 Function Tools（比如获取天气、计算器等工具）
                tool_list = load_all_tools( # J：加载所有工具（Native 工具和 Function 工具），这里的工具是提前定义的，不是从数据中动态加载的
                    tool_config_path=self.tool_config_path,
                    function_tool_path=self.function_tool_path,
                )
                self.tool_schemas = [ # J：将加载的每个工具（tool）的 schema（Pydantic 模型）序列化为 Python 字典
                    # J: tool.tool_schema 都是 OpenAIFunctionToolSchema 类型，排除 None 值和未设置的字段
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning(
                    "Failed to initialize tools (tool_config_path=%s, function_tool_path=%s): %s",
                    self.tool_config_path,
                    self.function_tool_path,
                    e,
                )
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)) # J：过滤过长提示的线程数，默认 CPU/4 个（至少 1 个）
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None # J：限制线程数不超过 CPU 核心数
        self.use_shm = config.get("use_shm", False) # J：是否使用共享内存缓存，默认 False
        self.chat_template_func = config.get("chat_template_func", None) # J：聊天模板函数，默认 None
        self.need_tools_kwargs = config.get("need_tools_kwargs", False) # J：是否需要工具参数，默认 False
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False) # J：默认不随机打乱数据集，默认 False
        self.seed = config.get("seed")

        self._download() # J：从 HDFS 下载数据文件到本地缓存目录
        self._read_files_and_tokenize() # J：读取数据文件，不进行分词，后续实现的 __getitem__ 中也不分词，apply_chat_template 和 Tokenize 会在训练循环中调用

    def _download(self, use_origin_parquet=False): # J：下载数据文件到本地缓存目录
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            # J：复制 HDFS 文件或目录到本地缓存目录
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self): # J：读取数据文件，不进行分词（后续实现的 __getitem__ 中也不分词），apply_chat_template 和 Tokenize 会在训练循环中调用
        dataframes = []
        for parquet_file in self.data_files:
            # read files and cache
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"] # J：读取 Parquet 文件
            elif parquet_file.endswith(".json") or parquet_file.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"] # J：读取 JSON 文件 
            else: # J：不支持的除 Parquet 和 JSON 文件以外的文件格式
                raise ValueError(f"Unsupported file format: {parquet_file}")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes) # J：将所有数据集合并为一个数据集

        total = len(self.dataframe) # J：数据集总样本数
        print(f"dataset len: {len(self.dataframe)}") # J：打印数据集总样本数

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args) # J：创建随机数生成器，np.random.default_rng() 是 NumPy 推荐的新一代随机数生成器（替代老旧的 np.random.seed() / np.random.rand()），基于 PCG64 随机算法，随机性更好、线程安全、可复现性强
                indices = rng.choice(total, size=self.max_samples, replace=False) # J：随机选择样本索引
            else:
                indices = np.arange(self.max_samples) # J：不需要 shuffle 时，直接选择前 self.max_samples 个样本的索引
            self.dataframe = self.dataframe.select(indices.tolist()) # J：根据索引选择数据集中的样本
            print(f"selected {self.max_samples} random samples out of {total}") # J：打印选择的样本数

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe) # J：过滤过长的 Prompt，仅在 filter_overlong_prompts 为 True 时进行过滤

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts: # J：是否过滤过长的 Prompt，不需要过滤时直接返回原始数据集
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key

            if processor is not None: # J：多模态的 doc2len 函数

                def doc2len(doc) -> int:
                    try:
                        # J：doc 是单行数据，可以是多轮
                        # J：原始的 doc 是一个包含多模态信息的字典，不是可以直接用于 processor.apply_chat_template 的消息列表
                        # J：注意：原始的 doc 中的占位符（如 <image>、<video>、<audio>）需要在生成消息列表时被替换为实际的图片、视频、音频等内容，否则会导致 processor.apply_chat_template 报错或生成无效的 Prompt 输入
                        messages = self._build_messages(doc, key=self.prompt_key) # J：生成可供 processor.apply_chat_template 处理的消息列表，包含图片、视频、音频模态
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template( # J：应用处理器的聊天模板，生成原始的 Prompt
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )
                        images, videos, audios = self._process_multi_modal_info( # J：处理多模态信息，提取图片、视频、音频
                            messages, self.image_patch_size, self.config
                        )
                        if images is None and videos is None and audios is None: # J：文本模态
                            # only text prompt
                            return len(
                                processor.tokenizer( # J：对文本进行分词，返回 input_ids
                                    text=raw_prompt,
                                    add_special_tokens=False,  # avoid adding special tokens
                                    return_attention_mask=False,
                                )["input_ids"]
                            )
                        else: # J：多模态
                            # multi-modal prompt
                            return len(
                                build_multimodal_processor_inputs( # J：构建多模态的编码结果，对多模态进行分词，返回 input_ids
                                    processor,
                                    text=[raw_prompt],
                                    images=images,
                                    videos=videos,
                                    audio=audios,
                                    mm_processor_kwargs=self.mm_processor_kwargs,
                                )["input_ids"][0]
                            )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else: # J：文本模态，单独定义 doc2len 函数，避免与多模态的 doc2len 函数冲突


                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs) # J：apply_chat_template 的参数
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        # Keep explicit tokenization to avoid transformers version default changes.
                        apply_kwargs.pop("tokenize", None) # J：默认进行分词，下面强制改为 True
                        apply_kwargs.pop("return_dict", None) # J：默认不返回字典格式的输出，默认 False
                        apply_kwargs.pop("return_tensors", None) # J：默认不返回张量格式的输出，默认 False

                        tokenized_prompt = tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, tokenize=True, **apply_kwargs
                        )
                        # J：将分词后的输出转换为平铺的 token ids 列表
                        return len(normalize_token_ids(tokenized_prompt))
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length, # J：过滤过长的 Prompt
                num_proc=self.num_workers, # J：并行处理，加速过滤
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens", # J：过滤描述
            )

            print(f"filter dataset len: {len(dataframe)}") # J：打印过滤后的数据集长度
        return dataframe # J：返回过滤后的数据集，格式仍为 datasets.Dataset 类型

    def resume_dataset_state(self): # J：恢复数据集的状态，从原始数据文件中读取数据并进行分词
        # J：已经初始化过是，original_data_files 属性是存在的，此时 serialize_dataset 为 False
        self.serialize_dataset = not hasattr(self, "original_data_files") # J：判断是否需要序列化数据集，注意这里是 not hasattr
        # resume dataframe if not it's serialized in data.pt
        # J：如果不需要序列化数据集，说明数据集是从原始数据文件中读取的（serialize_dataset 为 False）
        # J：如果需要序列化数据集，则重新走一遍初始化流程（serialize_dataset 为 True）
        if not self.serialize_dataset: 
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    def __len__(self):
        return len(self.dataframe)

    # J：example 是单行数据，可以包含多模态信息和多轮对话信息
    def _build_messages(self, example: dict, key: str): # J：生成可供 processor.apply_chat_template 处理的消息列表，包含图片、视频、音频模态
        """Replace multimodal placeholders in messages with structured content.

        This is required by processor.apply_chat_template.
        - <image>: {"type": "image", **image}
        - <video>: {"type": "video", **video}
        - <audio>: {"type": "audio", **audio}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        messages: list = example[key]
        # When concatenating multimodal datasets, get will return None for samples without a modality column.
        images = example.get(self.image_key, None) or []
        videos = example.get(self.video_key, None) or []
        audios = example.get(self.audio_key, None) or []

        image_offset, video_offset, audio_offset = 0, 0, 0
        for message in messages: # J：遍历每个消息（因为可能是多轮对话）
            if not images and not videos and not audios:
                continue
            assert self.processor is not None, "processor is needed to process multimodal data"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>|<audio>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>": # J：处理图片模态
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                        content_list.append({"type": "image", "image": image})
                    elif isinstance(image, dict):
                        if "bytes" in image:
                            image["image"] = Image.open(BytesIO(image["bytes"]))
                        content_list.append({"type": "image", **image})
                    else:
                        raise TypeError(f"image must be dict or PIL.Image, unsupported image type: {type(image)}")
                    image_offset += 1
                elif segment == "<video>": # J：处理视频模态
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    content_list.append({"type": "video", **videos[video_offset]})
                    video_offset += 1
                elif segment == "<audio>": # J：处理音频模态
                    assert audio_offset < len(audios), f"audio_offset {audio_offset} >= len(audios) {len(audios)}"
                    audio = audios[audio_offset]
                    if isinstance(audio, dict):
                        payload = dict(audio)
                        payload["type"] = "audio"
                        if "audio" not in payload and "audio_url" not in payload:
                            payload = {"type": "audio", "audio": audio}
                        content_list.append(payload)
                    else:
                        content_list.append({"type": "audio", "audio": audio})
                    audio_offset += 1
                else: # J：处理文本模态
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        # J：检查所有模态都已处理且没有剩余
        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        assert audio_offset == len(audios), f"audio_offset {audio_offset} != len(audios) {len(audios)}"
        return messages

    def __getitem__(self, item): # J：获取数据集中的一个样本，不进行分词，apply_chat_template 和 Tokenize 会在训练循环中调用
        """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
        row_dict: dict = self.dataframe[item] # J：抽取一行数据，可能包含 extra_info 字段
        row_dict["raw_prompt"] = self._build_messages(row_dict, key=self.prompt_key) # J：生成可供 processor.apply_chat_template 处理的消息列表，包含图片、视频、音频模态

        # J：删除原始数据中的图片、视频、音频模态，因为接下来他们没用了？（ processor.apply_chat_template 只需要消息列表）
        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)
        row_dict.pop(self.audio_key, None)

        # TODO(wuxibin): We still need a dummy tensor to make sure DataProto.batch is not empty.
        # Remove this after deprecate DataProto by TensorDict.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0) # J：获取数据的索引，默认值为 0，这里可以用于定位到原始数据
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        """Extract images and videos from messages.

        This method is called by AgentLoop (e.g SingleTurnAgentLoop) before apply_chat_template to
        the `raw_prompt` from dataset. User may customize RLHFDataset and override this method to
        support custom vision extraction.

        >>> messages = kwargs["raw_prompt"]
        >>> images, videos = RLHFDataset.process_vision_info(messages, image_patch_size)
        >>> videos, video_metadatas = zip(*videos)
        >>> raw_prompt = processor.apply_chat_template(messages, tokenize=False)
        >>> inputs = processor(text=[raw_prompt], images=images, videos=videos,
        ...                    video_metadata=video_metadatas, do_sample_frames=False)

        Args:
            messages: List of messages from dataset `raw_prompt`.
            image_patch_size: Image patch size for processor.
            config: Config for dataset.

        Returns:
            images: List of images.
            videos: List of videos, each video is a tuple of (video_tensor, video_metadata).
        """
        from qwen_vl_utils import process_vision_info # J: qwen_vl_utils 是个 pip 包

        # J：url 等资源需要在 process_vision_info 中处理，否则会导致 processor.apply_chat_template 报错或生成无效的 Prompt 输入
        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        return images, videos

    @classmethod
    def _extract_audio_info(cls, messages: list[dict]) -> list[Any]:
        audios = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "audio":
                    continue
                if "audio" in item:
                    audios.append(item["audio"])
                elif "audio_url" in item:
                    audios.append(item["audio_url"])
                else:
                    audios.append({k: v for k, v in item.items() if k != "type"})
        return audios or None

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[Any], list[Any]]:
        has_visual = any(
            isinstance(message.get("content"), list)
            and any(isinstance(item, dict) and item.get("type") in {"image", "video"} for item in message["content"])
            for message in messages
        )
        if has_visual:
            from qwen_vl_utils import process_vision_info # J: qwen_vl_utils 是个 pip 包

            # J：url 等资源需要在 process_vision_info 中处理，否则会导致 processor.apply_chat_template 报错或生成无效的 Prompt 输入
            # J：它的主要作用是从对话消息中提取图像或视频信息，并进行相应的预处理，以便后续送入视觉编码器（ViT）
            images, videos = process_vision_info( # J：return_video_metadata=True 时，videos 是一个元组列表，每个元组包含 (video_tensor, video_metadata)
                messages, image_patch_size=image_patch_size, return_video_metadata=True
            )
        else:
            images, videos = None, None
        audios = cls._extract_audio_info(messages) # J：从消息列表提取音频
        return images, videos, audios

    @classmethod
    async def process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[Any], list[Any]]:
        return cls._process_multi_modal_info(messages, image_patch_size=image_patch_size, config=config)

    def split(self, num_splits: int):
        """
        split the dataset into num_splits sub-datasets
        Args:
            num_splits: specified number of splits
        Returns:
            List[RLHFDataset]: list of RLHFDataset splits
        Raises:
            ValueError: if num_splits is not a positive integer
        """
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError(f"num_splits must be a positive integer, got {num_splits}")

        if not hasattr(self, "dataframe"):
            raise AttributeError(
                "dataframe not found in RLHFDataset\n"
                "reason: _read_files_and_tokenize() not called or Parquet file loading failed"
            )
        if self.dataframe is None:
            raise ValueError("RLHFDataset dataframe 为 None!")

        total_samples = len(self.dataframe)
        print(f"total_samples: {total_samples}")
        if total_samples == 0:
            raise ValueError("Cannot split an empty dataset")

        # Calculate effective sample count after dropping remainders if needed
        if total_samples % num_splits != 0:
            total_samples = total_samples - (total_samples % num_splits)
            logging.warning(f"Dropping {len(self.dataframe) % num_splits} samples, effective samples: {total_samples}")

        split_size = total_samples // num_splits
        splits = []

        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_splits - 1 else total_samples

            split_dataframe = self.dataframe.select(range(start_idx, end_idx))

            split_dataset = RLHFDataset(
                data_files=self.data_files,
                tokenizer=self.tokenizer,
                config=self.config,
                processor=self.processor,
                max_samples=self.max_samples,
            )
            split_dataset.dataframe = split_dataframe
            split_dataset.serialize_dataset = self.serialize_dataset
            split_dataset.original_data_files = self.original_data_files

            splits.append(split_dataset)

        return splits


def get_dataset_class(data_config: DictConfig): # J：根据配置获取 RLHFDataset 类，注：优先读取可以自定义的类 custom_cls.path 配置的类
    """Get RLHF dataset class.

    Args:
        data_config: The data config.

    Returns:
        dataset_cls: The dataset class.
    """

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        # J：根据配置加载自定义数据集类
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        # J：检查自定义数据集类是否继承 torch.utils.data.Dataset 类(自定义数据集类必须继承 torch.utils.data.Dataset 类)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset # J：默认使用 RLHFDataset 类
    print(f"Using dataset class: {dataset_cls.__name__}")

    return dataset_cls
