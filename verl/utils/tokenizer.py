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
"""Utils for tokenization."""

import types
import warnings

__all__ = [
    "hf_tokenizer",
    "hf_processor",
    "normalize_token_ids",
    "build_multimodal_processor_inputs",
    "get_processor_token_id",
]


def normalize_token_ids(tokenized_output) -> list[int]: # J：将各种格式的分词后的输出转换为平铺的 token ids 列表
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for idx, token_id in enumerate(token_ids):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        try:
            normalized_ids.append(int(token_id))
        except (TypeError, ValueError) as e:
            raise TypeError(f"token_id must be int-convertible, got {type(token_id).__name__}: {token_id!r}") from e
    return normalized_ids


def get_processor_token_id(processor, token_name: str) -> int | None:
    """Resolve a multimodal special token id from a processor.

    Newer processors may expose ``image_token``/``video_token`` strings instead
    of ``image_token_id``/``video_token_id`` integers. Fall back to tokenizer
    conversion so rollout code can stay processor-agnostic.
    """

    if processor is None:
        return None

    token_id_attr = f"{token_name}_token_id"
    token_id = getattr(processor, token_id_attr, None)
    if token_id is not None:
        return int(token_id)

    token_attr = f"{token_name}_token"
    token = getattr(processor, token_attr, None)
    tokenizer = getattr(processor, "tokenizer", None)
    if token is not None and tokenizer is not None:
        converted = tokenizer.convert_tokens_to_ids(token)
        if converted is not None:
            return int(converted)

    return None


def _split_videos_and_metadata(videos):
    if videos is None:
        return None, None
    videos = list(videos)
    if len(videos) > 0 and isinstance(videos[0], tuple):
        video_values, video_metadata = zip(*videos, strict=False)
        return list(video_values), list(video_metadata)
    return videos, None


def build_multimodal_processor_inputs(
    processor,
    *,
    text,
    images=None,
    videos=None,
    audio=None,
    mm_processor_kwargs=None,
    return_tensors: str = "pt",
):
    """Build kwargs for multimodal processor calls.

    This keeps the existing VL flow intact while extending it with audio-aware
    paths for processors that accept audio inputs.
    """
    processor_kwargs = dict(mm_processor_kwargs or {})
    if audio is not None and "sampling_rate" not in processor_kwargs:
        sampling_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
        if sampling_rate is not None:
            processor_kwargs["sampling_rate"] = int(sampling_rate)

    videos, video_metadata = _split_videos_and_metadata(videos)
    processor_kwargs.setdefault("return_tensors", return_tensors)

    if video_metadata is not None:
        processor_kwargs.setdefault("video_metadata", video_metadata)
        processor_kwargs.setdefault("do_sample_frames", False)

    processor_inputs = {"text": text, "images": images, "videos": videos, **processor_kwargs}
    if audio is not None:
        processor_inputs["audio"] = audio

    return processor(**processor_inputs)


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        Optional[transformers.ProcessorMixin]: The pretrained multimodal processor.
        Returns ``None`` for text-only models (including AutoProcessor fallbacks to
        tokenizer backends such as ``TokenizersBackend``).
    """
    from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

    try:
        # J: 如果不是多模态，这里会返回什么？
        # J：AutoProcessor.from_pretrained 只针对多模态模型设计；加载普通单模态模型（纯文本 / 纯图像分类等无 Processor 的模型）时，会返回纯文本模型的 tokenizer 实例
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        # In newer transformers, AutoProcessor may legitimately fall back to a
        # tokenizer backend (e.g. TokenizersBackend) for text-only models.
        # Treat it as "no multimodal processor" and let callers use hf_tokenizer.
        # J: 如果 processor 是 纯文本的实例(TokenizersBackend(PreTrainedTokenizerBase))，说明是纯文本模型，直接返回 None
        if isinstance(processor, PreTrainedTokenizerBase): # J: 判断，如果 processor 是 PreTrainedTokenizerBase 的实例（即纯文本模型得实例，此时回退到 tokenizer），直接返回 None
            return None

        # J: 加载模型配置，来自 config.json 文件，还会添加一些其他配置文件
        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        # Bind vlm model's get_rope_index method to processor.
        processor.config = config # J: 将读取到的 config 绑定模型配置到 processor 上（注：这里 processor 中可能不存在 config 属性，必须要绑定一下）
        # J: 根据 processor 类型，导入对应的模型 类，用于绑定 get_rope_index 和 get_vision_position_ids 到 processor
        model_class = None
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                model_class = Qwen2VLModel
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                model_class = Qwen2_5_VLModel
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                model_class = Qwen3VLModel
            case "Glm4vImageProcessor":
                from transformers.models.glm4v import Glm4vModel

                model_class = Glm4vModel
            case "MllamaProcessor":
                pass  # MllamaProcessor and MllamaModel doesn't have get_rope_index property
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")

        if model_class is not None:
            # J: 绑定 get_rope_index 方法到 processor
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            # J: 绑定 get_vision_position_ids 方法到 processor
            if hasattr(model_class, "get_vision_position_ids"):
                processor.get_vision_position_ids = types.MethodType(model_class.get_vision_position_ids, processor)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
