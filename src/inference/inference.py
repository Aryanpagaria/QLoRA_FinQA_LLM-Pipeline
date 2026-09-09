from __future__ import annotations
import copy
import random
import time
from pathlib import Path
from typing import Any
from collections.abc import Mapping
import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from src.utils.config_loader import load_configs


InferenceConfig = dict[str, Any]
GenerationResult = dict[str, Any]


def _load_inference_config() -> InferenceConfig:
    """
    Load and validate the complete inference configuration.
    """

    config = load_configs()

    required_top_level_keys = {
        "model",
        "lora",
        "tokenizer",
        "checkpoint",
        "inference",
    }

    missing_keys = sorted(
        required_top_level_keys
        - set(config.keys())
    )

    if missing_keys:
        raise RuntimeError(
            "Inference configuration is missing required "
            f"top-level sections: {missing_keys}"
        )

    inference_config = config["inference"]

    if not isinstance(
        inference_config,
        dict,
    ):
        raise RuntimeError(
            "Inference configuration must be a dictionary."
        )

    required_inference_keys = {
        "enabled",
        "generation",
        "chat",
        "output",
        "reproducibility",
    }

    missing_inference_keys = sorted(
        required_inference_keys
        - set(inference_config.keys())
    )

    if missing_inference_keys:
        raise RuntimeError(
            "Inference configuration is missing required "
            f"keys: {missing_inference_keys}"
        )

    _validate_inference_config(
        config
    )

    return config


def _validate_inference_config(
    config: InferenceConfig,
) -> None:
    """
    Validate every inference setting required by the
    production inference pipeline.
    """

    model_config = config["model"]

    if not isinstance(
        model_config,
        dict,
    ):
        raise TypeError(
            "model configuration must be a dictionary."
        )

    model_name = model_config.get(
        "name"
    )

    if not isinstance(
        model_name,
        str,
    ) or not model_name.strip():
        raise ValueError(
            "model.name must be a non-empty string."
        )

    quantization = model_config.get(
        "quantization"
    )

    if not isinstance(
        quantization,
        dict,
    ):
        raise TypeError(
            "model.quantization must be a dictionary."
        )

    if not isinstance(
        quantization.get(
            "enabled"
        ),
        bool,
    ):
        raise TypeError(
            "model.quantization.enabled must be boolean."
        )

    if not isinstance(
        quantization.get(
            "load_in_4bit"
        ),
        bool,
    ):
        raise TypeError(
            "model.quantization.load_in_4bit must be boolean."
        )

    if quantization["enabled"] and not torch.cuda.is_available():
        if quantization["load_in_4bit"]:
            raise RuntimeError(
                "4-bit inference requires CUDA, but CUDA "
                "is not available."
            )

    lora_config = config["lora"]

    if not isinstance(
        lora_config,
        dict,
    ):
        raise TypeError(
            "lora configuration must be a dictionary."
        )

    if not isinstance(
        lora_config.get(
            "enabled"
        ),
        bool,
    ):
        raise TypeError(
            "lora.enabled must be boolean."
        )

    if not lora_config["enabled"]:
        raise RuntimeError(
            "Inference requires the LoRA adapter to be enabled."
        )

    tokenizer_config = config["tokenizer"]

    if not isinstance(
        tokenizer_config,
        dict,
    ):
        raise TypeError(
            "tokenizer configuration must be a dictionary."
        )

    checkpoint_config = config["checkpoint"]

    if not isinstance(
        checkpoint_config,
        dict,
    ):
        raise TypeError(
            "checkpoint configuration must be a dictionary."
        )

    adapter_directory = checkpoint_config.get(
        "lora_export_directory"
    )

    if not isinstance(
        adapter_directory,
        str,
    ) or not adapter_directory.strip():
        raise ValueError(
            "checkpoint.lora_export_directory must be "
            "a non-empty string."
        )

    inference_config = config["inference"]

    enabled = inference_config["enabled"]

    if not isinstance(
        enabled,
        bool,
    ):
        raise TypeError(
            "inference.enabled must be boolean."
        )

    generation = inference_config[
        "generation"
    ]

    if not isinstance(
        generation,
        dict,
    ):
        raise TypeError(
            "inference.generation must be a dictionary."
        )

    required_generation_keys = {
        "max_new_tokens",
        "temperature",
        "do_sample",
        "top_p",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "early_stopping",
        "use_cache",
    }

    missing_generation_keys = sorted(
        required_generation_keys
        - set(generation.keys())
    )

    if missing_generation_keys:
        raise RuntimeError(
            "Missing generation configuration keys: "
            f"{missing_generation_keys}"
        )

    max_new_tokens = generation[
        "max_new_tokens"
    ]

    if not isinstance(
        max_new_tokens,
        int,
    ) or max_new_tokens <= 0:
        raise ValueError(
            "generation.max_new_tokens must be "
            "a positive integer."
        )

    temperature = generation[
        "temperature"
    ]

    if not isinstance(
        temperature,
        (int, float),
    ) or temperature < 0:
        raise ValueError(
            "generation.temperature must be "
            "greater than or equal to zero."
        )

    top_p = generation[
        "top_p"
    ]

    if not isinstance(
        top_p,
        (int, float),
    ) or not 0 < top_p <= 1:
        raise ValueError(
            "generation.top_p must be in the range (0, 1]."
        )

    repetition_penalty = generation[
        "repetition_penalty"
    ]

    if not isinstance(
        repetition_penalty,
        (int, float),
    ) or repetition_penalty <= 0:
        raise ValueError(
            "generation.repetition_penalty must be "
            "greater than zero."
        )

    no_repeat_ngram_size = generation[
        "no_repeat_ngram_size"
    ]

    if not isinstance(
        no_repeat_ngram_size,
        int,
    ) or no_repeat_ngram_size < 0:
        raise ValueError(
            "generation.no_repeat_ngram_size must be "
            "a non-negative integer."
        )

    boolean_generation_keys = {
        "do_sample",
        "early_stopping",
        "use_cache",
    }

    for key in boolean_generation_keys:

        if not isinstance(
            generation[key],
            bool,
        ):
            raise TypeError(
                f"generation.{key} must be boolean."
            )

    if (
        not generation["do_sample"]
        and temperature != 0
    ):
        raise ValueError(
            "Deterministic inference requires "
            "temperature=0 when do_sample=false."
        )

    chat_config = inference_config[
        "chat"
    ]

    if not isinstance(
        chat_config,
        dict,
    ):
        raise TypeError(
            "inference.chat must be a dictionary."
        )

    max_history = chat_config.get(
        "max_history"
    )

    if not isinstance(
        max_history,
        int,
    ) or max_history <= 0:
        raise ValueError(
            "chat.max_history must be a positive integer."
        )

    exit_commands = chat_config.get(
        "exit_commands"
    )

    if not isinstance(
        exit_commands,
        list,
    ) or not exit_commands:
        raise ValueError(
            "chat.exit_commands must be a non-empty list."
        )

    if not all(
        isinstance(
            command,
            str,
        ) and command.strip()
        for command in exit_commands
    ):
        raise ValueError(
            "Every chat exit command must be a "
            "non-empty string."
        )

    output_config = inference_config[
        "output"
    ]

    if not isinstance(
        output_config,
        dict,
    ):
        raise TypeError(
            "inference.output must be a dictionary."
        )

    if not isinstance(
        output_config.get(
            "save_responses"
        ),
        bool,
    ):
        raise TypeError(
            "output.save_responses must be boolean."
        )

    output_directory = output_config.get(
        "directory"
    )

    if not isinstance(
        output_directory,
        str,
    ) or not output_directory.strip():
        raise ValueError(
            "output.directory must be a non-empty string."
        )

    reproducibility = inference_config[
        "reproducibility"
    ]

    if not isinstance(
        reproducibility,
        dict,
    ):
        raise TypeError(
            "inference.reproducibility must be a dictionary."
        )

    seed = reproducibility.get(
        "seed"
    )

    if not isinstance(
        seed,
        int,
    ) or seed < 0:
        raise ValueError(
            "reproducibility.seed must be a "
            "non-negative integer."
        )


def _set_reproducibility_seed(
    seed: int,
) -> None:
    """
    Set all supported random-number generators to the
    configured deterministic seed.
    """

    if not isinstance(
        seed,
        int,
    ) or seed < 0:
        raise ValueError(
            "seed must be a non-negative integer."
        )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def _resolve_adapter_directory(
    config: InferenceConfig,
) -> Path:
    """
    Resolve the exported LoRA adapter directory.
    """

    checkpoint_config = config[
        "checkpoint"
    ]

    configured_directory = checkpoint_config[
        "lora_export_directory"
    ]

    adapter_directory = Path(
        configured_directory
    )

    if not adapter_directory.exists():
        raise FileNotFoundError(
            "LoRA adapter directory does not exist: "
            f"{adapter_directory}"
        )

    if not adapter_directory.is_dir():
        raise RuntimeError(
            "LoRA adapter path is not a directory: "
            f"{adapter_directory}"
        )

    required_files = {
        "adapter_config.json",
    }

    missing_files = sorted(
        file_name
        for file_name in required_files
        if not (
            adapter_directory
            / file_name
        ).is_file()
    )

    adapter_weight_files = [
        path
        for path in adapter_directory.iterdir()
        if path.is_file()
        and path.name in {
            "adapter_model.safetensors",
            "adapter_model.bin",
        }
    ]

    if not adapter_weight_files:
        missing_files.append(
            "adapter_model.safetensors or adapter_model.bin"
        )

    if missing_files:
        raise FileNotFoundError(
            "LoRA adapter directory is incomplete. "
            f"Missing: {missing_files}. "
            f"Directory: {adapter_directory}"
        )

    return adapter_directory


def _load_tokenizer(
    config: InferenceConfig,
) -> PreTrainedTokenizerBase:
    """
    Load the production tokenizer from the configured base model.
    """

    model_config = config[
        "model"
    ]

    tokenizer_config = config[
        "tokenizer"
    ]

    model_name = model_config[
        "name"
    ]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(
                model_config.get(
                    "trust_remote_code",
                    False,
                )
            ),
            use_fast=bool(
                tokenizer_config.get(
                    "use_fast",
                    True,
                )
            ),
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to load tokenizer for inference "
            f"from '{model_name}'."
        ) from error

    if not isinstance(
        tokenizer,
        PreTrainedTokenizerBase,
    ):
        raise TypeError(
            "Loaded tokenizer is not a "
            "PreTrainedTokenizerBase."
        )

    padding_side = tokenizer_config.get(
        "padding_side"
    )

    truncation_side = tokenizer_config.get(
        "truncation_side"
    )

    if padding_side not in {
        "left",
        "right",
    }:
        raise ValueError(
            "tokenizer.padding_side must be "
            "'left' or 'right'."
        )

    if truncation_side not in {
        "left",
        "right",
    }:
        raise ValueError(
            "tokenizer.truncation_side must be "
            "'left' or 'right'."
        )

    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side

    if tokenizer.pad_token is None:

        if tokenizer.eos_token is None:
            raise RuntimeError(
                "Tokenizer has neither a pad token "
                "nor an EOS token."
            )

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    if tokenizer.eos_token_id is None:
        raise RuntimeError(
            "Tokenizer does not define an EOS token ID."
        )

    print("=" * 80)
    print("TOKENIZER READY")
    print("=" * 80)

    print(
        f"Tokenizer       : "
        f"{tokenizer.__class__.__name__}"
    )

    print(
        f"Vocabulary      : "
        f"{len(tokenizer):,}"
    )

    print(
        f"Padding Side    : "
        f"{tokenizer.padding_side}"
    )

    print(
        f"Pad Token       : "
        f"{tokenizer.pad_token}"
    )

    print(
        f"EOS Token       : "
        f"{tokenizer.eos_token}"
    )

    print("=" * 80)

    return tokenizer


def _build_quantization_config(
    config: InferenceConfig,
) -> BitsAndBytesConfig | None:
    """
    Build the bitsandbytes quantization configuration.
    """

    quantization = config[
        "model"
    ][
        "quantization"
    ]

    if not quantization[
        "enabled"
    ]:
        return None

    if not quantization[
        "load_in_4bit"
    ]:
        return None

    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit QLoRA inference requires CUDA."
        )

    compute_dtype_name = quantization[
        "compute_dtype"
    ]

    if compute_dtype_name == "float16":

        compute_dtype = torch.float16

    elif compute_dtype_name == "bfloat16":

        compute_dtype = torch.bfloat16

    elif compute_dtype_name == "float32":

        compute_dtype = torch.float32

    else:

        raise ValueError(
            "Unsupported quantization compute dtype: "
            f"{compute_dtype_name}"
        )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization[
            "quantization_type"
        ],
        bnb_4bit_use_double_quant=quantization[
            "use_double_quantization"
        ],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _load_base_model(
    config: InferenceConfig,
) -> torch.nn.Module:
    """
    Load the frozen quantized base causal language model.
    """

    model_config = config[
        "model"
    ]

    model_name = model_config[
        "name"
    ]

    quantization_config = (
        _build_quantization_config(
            config
        )
    )

    device_config = model_config.get(
        "device",
        {},
    )

    if not isinstance(
        device_config,
        dict,
    ):
        raise TypeError(
            "model.device must be a dictionary."
        )

    device_map = device_config.get(
        "device_map",
        "auto",
    )

    if torch.cuda.is_available():

        if device_map != "auto":
            raise ValueError(
                "Inference currently requires "
                "device_map='auto' when CUDA is available."
            )

    else:

        device_map = None

    try:

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=bool(
                model_config.get(
                    "trust_remote_code",
                    False,
                )
            ),
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to load base model for inference: "
            f"{model_name}"
        ) from error

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "Loaded base model is not a torch.nn.Module."
        )

    model.eval()

    if hasattr(
        model,
        "config",
    ):
        model.config.use_cache = True

    for parameter in model.parameters():
        parameter.requires_grad = False

    try:

        first_parameter = next(
            model.parameters()
        )

    except StopIteration as error:

        raise RuntimeError(
            "Loaded base model contains no parameters."
        ) from error

    print("=" * 80)
    print("BASE MODEL READY")
    print("=" * 80)

    print(
        f"Base Model      : "
        f"{model_name}"
    )

    print(
        f"Device          : "
        f"{first_parameter.device}"
    )

    print(
        f"Quantized       : "
        f"{quantization_config is not None}"
    )

    print("=" * 80)

    return model


def _load_lora_adapter(
    model: torch.nn.Module,
    config: InferenceConfig,
) -> PeftModel:
    """
    Attach the exported LoRA adapter to the frozen base model.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    adapter_directory = (
        _resolve_adapter_directory(
            config
        )
    )

    try:

        peft_model = PeftModel.from_pretrained(
            model,
            str(
                adapter_directory
            ),
            is_trainable=False,
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to load the LoRA adapter from "
            f"{adapter_directory}"
        ) from error

    if not isinstance(
        peft_model,
        PeftModel,
    ):
        raise TypeError(
            "Loaded adapter model is not a PeftModel."
        )

    peft_model.eval()

    trainable_parameters = [
        parameter
        for parameter in peft_model.parameters()
        if parameter.requires_grad
    ]

    if trainable_parameters:
        raise RuntimeError(
            "Inference model unexpectedly contains "
            "trainable parameters."
        )

    print("=" * 80)
    print("LORA ADAPTER READY")
    print("=" * 80)

    print(
        f"Adapter         : "
        f"{adapter_directory}"
    )

    print(
        "Trainable Params: 0"
    )

    print("=" * 80)

    return peft_model


def _build_messages(
    history: list[dict[str, str]],
    user_prompt: str,
    system_message: str,
) -> list[dict[str, str]]:
    """
    Build a validated Qwen chat conversation.
    """

    if not isinstance(
        history,
        list,
    ):
        raise TypeError(
            "history must be a list."
        )

    if not isinstance(
        user_prompt,
        str,
    ):
        raise TypeError(
            "user_prompt must be a string."
        )

    user_prompt = user_prompt.strip()

    if not user_prompt:
        raise ValueError(
            "user_prompt cannot be empty."
        )

    if not isinstance(
        system_message,
        str,
    ) or not system_message.strip():
        raise ValueError(
            "system_message must be a non-empty string."
        )

    new_history = copy.deepcopy(
        history
    )

    if not new_history:

        new_history.append(
            {
                "role": "system",
                "content": system_message.strip(),
            }
        )

    new_history.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    return new_history


def _build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
) -> str:
    """
    Convert validated chat messages into the tokenizer's
    native chat-template prompt.
    """

   

    if not isinstance(
        messages,
        list,
    ) or not messages:
        raise ValueError(
            "messages must be a non-empty list."
        )

    if not hasattr(
        tokenizer,
        "apply_chat_template",
    ):
        raise RuntimeError(
            "Tokenizer does not provide apply_chat_template."
        )

    try:

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to construct the model chat prompt."
        ) from error

    if not isinstance(
        prompt,
        str,
    ) or not prompt.strip():
        raise RuntimeError(
            "Tokenizer produced an empty inference prompt."
        )

    return prompt

def _prepare_generation_inputs(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and move all tensor inputs to the
    model execution device.
    """

    if not isinstance(
        tokenizer,
        PreTrainedTokenizerBase,
    ):
        raise TypeError(
            "tokenizer must be a "
            "PreTrainedTokenizerBase."
        )

    if not isinstance(
        prompt,
        str,
    ) or not prompt.strip():
        raise ValueError(
            "prompt must be a non-empty string."
        )

    if not isinstance(
        device,
        torch.device,
    ):
        raise TypeError(
            "device must be a torch.device."
        )

    try:

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to tokenize the inference prompt."
        ) from error

    if not isinstance(
        encoded,
        Mapping,
    ):
        raise TypeError(
            "Tokenizer output must implement the "
            "mapping interface."
        )

    prepared_inputs: dict[
        str,
        torch.Tensor,
    ] = {}

    for key, value in encoded.items():

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"Tokenizer output '{key}' must be a tensor."
            )

        prepared_inputs[key] = value.to(
            device=device
        )

    if "input_ids" not in prepared_inputs:
        raise RuntimeError(
            "Tokenizer output does not contain input_ids."
        )

    if prepared_inputs[
        "input_ids"
    ].ndim != 2:
        raise RuntimeError(
            "input_ids must have shape [batch, sequence]."
        )

    if prepared_inputs[
        "input_ids"
    ].shape[0] != 1:
        raise RuntimeError(
            "Inference currently supports exactly "
            "one prompt at a time."
        )

    return prepared_inputs
def _build_generation_kwargs(
    config: InferenceConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """
    Construct deterministic or sampled generation arguments.
    """

    generation = config[
        "inference"
    ][
        "generation"
    ]

    generation_kwargs: dict[
        str,
        Any,
    ] = {
        "max_new_tokens": generation[
            "max_new_tokens"
        ],
        "do_sample": generation[
            "do_sample"
        ],
        "repetition_penalty": generation[
            "repetition_penalty"
        ],
        "no_repeat_ngram_size": generation[
            "no_repeat_ngram_size"
        ],
        "early_stopping": generation[
            "early_stopping"
        ],
        "use_cache": generation[
            "use_cache"
        ],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if generation[
        "do_sample"
    ]:

        generation_kwargs[
            "temperature"
        ] = generation[
            "temperature"
        ]

        generation_kwargs[
            "top_p"
        ] = generation[
            "top_p"
        ]

    return generation_kwargs


def generate_response(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    config: InferenceConfig,
) -> GenerationResult:
    """
    Generate one deterministic or sampled response.

    The returned dictionary contains the decoded response and
    reproducibility/performance metadata.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if not isinstance(
        tokenizer,
        PreTrainedTokenizerBase,
    ):
        raise TypeError(
            "tokenizer must be a "
            "PreTrainedTokenizerBase."
        )

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "config must be a dictionary."
        )

    model.eval()

    try:

        first_parameter = next(
            model.parameters()
        )

    except StopIteration as error:

        raise RuntimeError(
            "Model contains no parameters."
        ) from error

    device = first_parameter.device

    inputs = _prepare_generation_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
    )

    prompt_token_count = int(
        inputs[
            "input_ids"
        ].shape[-1]
    )

    generation_kwargs = _build_generation_kwargs(
        config=config,
        tokenizer=tokenizer,
    )

    start_time = time.perf_counter()

    with torch.inference_mode():

        try:

            output_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )

        except Exception as error:

            raise RuntimeError(
                "Model generation failed."
            ) from error

    end_time = time.perf_counter()

    if not isinstance(
        output_ids,
        torch.Tensor,
    ):
        raise TypeError(
            "model.generate() must return a torch.Tensor."
        )

    if output_ids.ndim != 2:
        raise RuntimeError(
            "Generated output must have shape "
            "[batch, sequence]."
        )

    if output_ids.shape[0] != 1:
        raise RuntimeError(
            "Inference expects exactly one generated sequence."
        )

    generated_token_ids = output_ids[
        0,
        prompt_token_count:,
    ]

    generated_token_count = int(
        generated_token_ids.shape[-1]
    )

    generation_time = (
        end_time
        - start_time
    )

    if generation_time <= 0:
        raise RuntimeError(
            "Generation completed with an invalid "
            "non-positive duration."
        )

    try:

        response = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    except Exception as error:

        raise RuntimeError(
            "Failed to decode generated tokens."
        ) from error

    if not response:
        raise RuntimeError(
            "Model generated an empty response."
        )

    tokens_per_second = (
        generated_token_count
        / generation_time
    )

    return {
        "response": response,
        "prompt_tokens": prompt_token_count,
        "generated_tokens": generated_token_count,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
        "temperature": config[
            "inference"
        ][
            "generation"
        ][
            "temperature"
        ],
        "top_p": config[
            "inference"
        ][
            "generation"
        ][
            "top_p"
        ],
        "do_sample": config[
            "inference"
        ][
            "generation"
        ][
            "do_sample"
        ],
    }


def _trim_history(
    history: list[dict[str, str]],
    max_history: int,
) -> list[dict[str, str]]:
    """
    Keep the system message plus the newest conversation turns.
    """

    if not isinstance(
        history,
        list,
    ):
        raise TypeError(
            "history must be a list."
        )

    if not isinstance(
        max_history,
        int,
    ) or max_history <= 0:
        raise ValueError(
            "max_history must be a positive integer."
        )

    if len(history) <= max_history:
        return history

    if history and history[0].get(
        "role"
    ) == "system":

        return [
            history[0],
            *history[
                -(max_history - 1):
            ],
        ]

    return history[
        -max_history:
    ]


def load_inference_stack() -> tuple[
    InferenceConfig,
    PreTrainedTokenizerBase,
    torch.nn.Module,
]:
    """
    Load the complete production inference stack once.

    This is the public entry point used by LocalProvider.

    The function:
    1. Loads and validates inference configuration.
    2. Sets the reproducibility seed.
    3. Loads the tokenizer.
    4. Loads the base Qwen model.
    5. Loads the exported LoRA adapter.
    6. Returns the ready-to-use inference components.

    Returns
    -------
    tuple
        (config, tokenizer, model)

    Raises
    ------
    RuntimeError
        If inference is disabled or any part of the inference stack
        cannot be loaded.
    """

    config = _load_inference_config()

    if not config["inference"]["enabled"]:
        raise RuntimeError(
            "Inference is disabled in configs/inference/inference.yaml."
        )

    _set_reproducibility_seed(config)

    tokenizer = _load_tokenizer(config)

    model = _load_base_model(config)

    model = _load_lora_adapter(
        model=model,
        config=config,
    )

    return config, tokenizer, model

class _TestTokenizer(PreTrainedTokenizerBase):
    """
    Minimal local tokenizer substitute used only by the isolated
    inference subsystem test.

    This tokenizer implements the minimum Hugging Face tokenizer
    contract required by the production inference functions while
    remaining completely local and deterministic.
    """

    model_input_names = [
        "input_ids",
        "attention_mask",
    ]

    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            pad_token=self.pad_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            **kwargs,
        )

    def __call__(
        self,
        prompt: str,
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:

        del padding
        del truncation
        del kwargs

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "Test tokenizer prompt must be a string."
            )

        if not prompt.strip():
            raise ValueError(
                "Test tokenizer received an empty prompt."
            )

        if return_tensors not in {
            None,
            "pt",
        }:
            raise ValueError(
                "Test tokenizer only supports "
                "return_tensors='pt'."
            )

        token_count = max(
            1,
            len(
                prompt.split()
            ),
        )

        input_ids = torch.ones(
            (
                1,
                token_count,
            ),
            dtype=torch.long,
        )

        attention_mask = torch.ones(
            (
                1,
                token_count,
            ),
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        **kwargs: Any,
    ) -> str:

        del skip_special_tokens
        del clean_up_tokenization_spaces
        del kwargs

        if not isinstance(
            token_ids,
            torch.Tensor,
        ):
            raise TypeError(
                "Test tokenizer token_ids must be a torch.Tensor."
            )

        if token_ids.numel() == 0:
            raise ValueError(
                "Test tokenizer cannot decode an empty tensor."
            )

        return "test response"

    def _convert_token_to_id(
        self,
        token: str,
    ) -> int:

        if token == self.pad_token:
            return self.pad_token_id

        if token == self.eos_token:
            return self.eos_token_id

        if token == self.unk_token:
            return self.unk_token_id

        return self.unk_token_id

    def _convert_id_to_token(
        self,
        index: int,
    ) -> str:

        if index == self.pad_token_id:
            return self.pad_token

        if index == self.eos_token_id:
            return self.eos_token

        if index == self.unk_token_id:
            return self.unk_token

        return self.unk_token

    def get_vocab(
        self,
    ) -> dict[str, int]:

        return {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id,
        }

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:

        if token_ids_1 is None:
            return list(token_ids_0)

        return [
            *token_ids_0,
            *token_ids_1,
        ]
class _TestGenerationModel(
    torch.nn.Module
):
    """
    Minimal deterministic causal-generation model used only
    for the isolated inference subsystem test.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.parameter = torch.nn.Parameter(
            torch.zeros(
                1,
                dtype=torch.float32,
            )
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:

        del attention_mask

        required_keys = {
            "max_new_tokens",
            "do_sample",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "early_stopping",
            "use_cache",
            "pad_token_id",
            "eos_token_id",
        }

        missing_keys = sorted(
            required_keys
            - set(kwargs.keys())
        )

        if missing_keys:
            raise RuntimeError(
                "Test generation call is missing required "
                f"generation arguments: {missing_keys}"
            )

        requested_tokens = int(
            kwargs[
                "max_new_tokens"
            ]
        )

        generated_count = min(
            3,
            requested_tokens,
        )

        generated = torch.ones(
            (
                1,
                generated_count,
            ),
            dtype=torch.long,
            device=input_ids.device,
        )

        return torch.cat(
            (
                input_ids,
                generated,
            ),
            dim=-1,
        )


def _test_configuration() -> None:
    """
    Test configuration loading and validation.
    """

    config = _load_inference_config()

    if not config[
        "inference"
    ][
        "enabled"
    ]:
        raise RuntimeError(
            "Inference must be enabled for the module test."
        )


def _test_prompt_construction() -> None:
    """
    Test prompt construction without downloading a model.
    """

    class TestChatTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:

            if tokenize:
                raise RuntimeError(
                    "Test requires tokenize=False."
                )

            if not add_generation_prompt:
                raise RuntimeError(
                    "Test requires add_generation_prompt=True."
                )

            return "\n".join(
                f"{message['role']}: "
                f"{message['content']}"
                for message in messages
            )

    tokenizer = TestChatTokenizer()

    messages = _build_messages(
        history=[],
        user_prompt="What is revenue?",
        system_message=(
            "You are a helpful financial assistant."
        ),
    )

    prompt = _build_prompt(
        tokenizer=tokenizer,
        messages=messages,
    )

    if "What is revenue?" not in prompt:
        raise RuntimeError(
            "Prompt construction did not preserve the user question."
        )

    if "system:" not in prompt:
        raise RuntimeError(
            "Prompt construction did not preserve the system message."
        )


def _test_generation_logic() -> None:
    """
    Test generation logic entirely locally without loading Qwen.
    """

    config = {
        "inference": {
            "generation": {
                "max_new_tokens": 8,
                "temperature": 0.0,
                "do_sample": False,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "early_stopping": True,
                "use_cache": True,
            }
        }
    }

    model = _TestGenerationModel()

    tokenizer = _TestTokenizer()

    prompt = (
        "system: You are a financial assistant.\n"
        "user: What is revenue?"
    )

    result = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
    )

    if result[
        "response"
    ] != "test response":
        raise RuntimeError(
            "Generation test returned an unexpected response."
        )

    if result[
        "prompt_tokens"
    ] <= 0:
        raise RuntimeError(
            "Generation test reported an invalid prompt token count."
        )

    if result[
        "generated_tokens"
    ] != 3:
        raise RuntimeError(
            "Generation test reported an unexpected "
            "generated token count."
        )

    if result[
        "generation_time"
    ] <= 0:
        raise RuntimeError(
            "Generation test reported an invalid generation time."
        )

    if result[
        "tokens_per_second"
    ] <= 0:
        raise RuntimeError(
            "Generation test reported an invalid throughput."
        )


def _test_history_management() -> None:
    """
    Test conversation-history retention.
    """

    history = [
        {
            "role": "system",
            "content": "system",
        },
        {
            "role": "user",
            "content": "one",
        },
        {
            "role": "assistant",
            "content": "two",
        },
        {
            "role": "user",
            "content": "three",
        },
        {
            "role": "assistant",
            "content": "four",
        },
    ]

    trimmed = _trim_history(
        history=history,
        max_history=3,
    )

    if len(trimmed) != 3:
        raise RuntimeError(
            "History trimming produced an incorrect length."
        )

    if trimmed[0][
        "role"
    ] != "system":
        raise RuntimeError(
            "History trimming removed the system message."
        )

    if trimmed[-1][
        "content"
    ] != "four":
        raise RuntimeError(
            "History trimming did not retain the newest message."
        )


def _test_reproducibility() -> None:
    """
    Verify deterministic seed restoration for the inference
    random-number generators.
    """

    seed = 42

    _set_reproducibility_seed(
        seed
    )

    python_first = random.random()
    numpy_first = float(
        np.random.random()
    )
    torch_first = torch.rand(
        1
    )

    _set_reproducibility_seed(
        seed
    )

    python_second = random.random()
    numpy_second = float(
        np.random.random()
    )
    torch_second = torch.rand(
        1
    )

    if python_first != python_second:
        raise RuntimeError(
            "Python inference RNG is not deterministic."
        )

    if numpy_first != numpy_second:
        raise RuntimeError(
            "NumPy inference RNG is not deterministic."
        )

    if not torch.equal(
        torch_first,
        torch_second,
    ):
        raise RuntimeError(
            "PyTorch inference RNG is not deterministic."
        )


def main() -> None:
    """
    Run the isolated inference subsystem test.

    The test intentionally does not load the 3B production model
    or require a trained LoRA adapter. Production model loading
    is exercised separately after the first real training run.
    """

    print("=" * 80)
    print("INFERENCE MODULE TEST")
    print("=" * 80)

    _test_configuration()

    print(
        "Configuration          : PASSED"
    )

    _test_prompt_construction()

    print(
        "Prompt Construction    : PASSED"
    )

    _test_generation_logic()

    print(
        "Generation Logic       : PASSED"
    )

    _test_history_management()

    print(
        "Chat History Logic     : PASSED"
    )

    _test_reproducibility()

    print(
        "Reproducibility        : PASSED"
    )

    print("=" * 80)
    print(
        "Inference module is ready for "
        "production model integration."
    )
    print(
        "Status                  : PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()