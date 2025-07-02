"""
Model loading utilities for InternVL Evaluation

This module handles loading the InternVL model and tokenizer.
Supports both absolute paths and paths relative to project root.
"""

import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from huggingface_hub.utils import HFValidationError
from transformers import AutoModel, AutoTokenizer

from internvl.utils.logging import get_logger
from internvl.utils.path import enforce_module_invocation

# Enforce module invocation pattern
enforce_module_invocation("internvl.model")

# Get logger for this module
logger = get_logger(__name__)


def suppress_tokenizer_warnings():
    """Suppress common tokenizer warnings that are safe to ignore during inference."""
    # Suppress the pad_token_id warning specifically
    warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")
    warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")
    warnings.filterwarnings("ignore", message=".*does not have a pad_token_id.*")
    
    # Suppress transformers library warnings
    try:
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass


def split_model(model_name: str) -> Dict[str, int]:
    """
    Create device mapping for multi-GPU configuration.
    Based on the model architecture, distributes layers across available GPUs.

    Args:
        model_name: Name of the model to determine layer count

    Returns:
        Dictionary mapping model components to GPU devices
    """
    device_map = {}
    world_size = torch.cuda.device_count()

    # Model layer counts for different InternVL models
    num_layers_mapping = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
        "InternVL3-8B": 28,
    }

    # Extract model size from path for layer count determination
    model_size = None
    for size_key in num_layers_mapping.keys():
        if (
            size_key in model_name
            or size_key.replace("-", "_").lower() in model_name.lower()
        ):
            model_size = size_key
            break

    if model_size is None:
        # Default to InternVL3-8B if we can't determine the model size
        logger.warning(
            f"Could not determine model size from {model_name}, defaulting to InternVL3-8B"
        )
        model_size = "InternVL3-8B"

    num_layers = num_layers_mapping[model_size]

    # Since the first GPU will be used for ViT, treat it as half a GPU
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1

    # Vision and core components on GPU 0
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def auto_detect_device_config() -> Tuple[str, int, bool]:
    """
    Automatically detect the best device configuration based on available hardware.

    Returns:
        Tuple of (device_type, num_gpus, use_quantization)
    """
    if not torch.cuda.is_available():
        return "cpu", 0, False

    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        return "cuda", 1, True  # Use 8-bit quantization for single GPU
    else:
        return "cuda", num_gpus, False  # Use multi-GPU without quantization


def load_model_and_tokenizer(
    model_path: str,
    device: Optional[str] = None,
    use_flash_attn: bool = False,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    auto_device_config: bool = True,
) -> Tuple[Any, Any]:
    """
    Load InternVL model and tokenizer with automatic device configuration.
    Supports both HuggingFace model IDs and local paths (absolute or relative).
    Automatically detects GPU configuration and applies optimal settings.

    Args:
        model_path: Path to model directory or HuggingFace model ID
        device: Device to load model on ('cuda', 'cpu', or None for auto-detection)
        use_flash_attn: Whether to use FlashAttention for faster inference
        trust_remote_code: Whether to trust remote code in model
        local_files_only: Whether to only use local files (for local models)
        auto_device_config: Whether to automatically configure device settings

    Returns:
        Tuple of (model, tokenizer)
    """
    # IMPORTANT: For models, we prioritize exact paths in environment variables
    # rather than trying to resolve them relative to project root
    env_model_path = os.environ.get("INTERNVL_MODEL_PATH")
    if env_model_path:
        # Use the environment variable directly since model paths are typically absolute
        # and may point to external volumes in KFP environments
        model_path = env_model_path
        logger.info(f"Using model path from environment variable: {model_path}")
    else:
        logger.info(f"Using provided model path: {model_path}")

    # Determine if it's a HuggingFace cache path (storage mapped checkpoint)
    path_obj = Path(model_path)
    is_hf_cache_path = "models--" in str(path_obj) and "hub" in str(path_obj)
    is_local_model = (
        path_obj.is_absolute()
        or model_path.startswith("./")
        or model_path.startswith("../")
    ) and Path(model_path).exists() and not is_hf_cache_path

    if is_hf_cache_path:
        # For HuggingFace cache paths, we need to use local_files_only=True
        # and keep the cache path as-is for direct loading
        logger.info(f"Detected HuggingFace cache path: {model_path}")
        logger.info("Using cached model files with local_files_only=True")
        local_files_only = True
    elif is_local_model:
        logger.info(f"Detected local model path: {model_path}")
        logger.info("Using local model files")
        local_files_only = True
    else:
        logger.info(f"Treating as HuggingFace model ID: {model_path}")
        local_files_only = False

    logger.info(f"Final model path for loading: {model_path}")

    # Auto-detect device configuration if not specified
    if auto_device_config and device is None:
        device_type, num_gpus, use_quantization = auto_detect_device_config()
        logger.info(
            f"Auto-detected configuration: {device_type}, {num_gpus} GPUs, quantization: {use_quantization}"
        )
    else:
        device_type = device if device else "cpu"
        num_gpus = torch.cuda.device_count() if device_type == "cuda" else 0
        use_quantization = False
        logger.info(f"Manual configuration: {device_type}, {num_gpus} GPUs")

    # Configure loading parameters based on device configuration
    model_loading_args = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }

    if local_files_only:
        model_loading_args["local_files_only"] = True
        logger.info("Setting local_files_only=True for model loading")

    # Configure based on device type and GPU count
    if device_type == "cpu":
        model_loading_args["torch_dtype"] = torch.float32
        logger.info("Loading model on CPU (will be slow)...")
    elif num_gpus == 1:
        model_loading_args["torch_dtype"] = torch.bfloat16
        if use_quantization:
            model_loading_args["load_in_8bit"] = True
            logger.info("Loading model on single GPU with 8-bit quantization...")
        else:
            logger.info("Loading model on single GPU...")
        model_loading_args["use_flash_attn"] = use_flash_attn
    else:  # Multi-GPU
        model_loading_args["torch_dtype"] = torch.bfloat16
        device_map = split_model(model_path)
        model_loading_args["device_map"] = device_map
        logger.info(f"Loading model across {num_gpus} GPUs...")
        logger.info(f"Device mapping: {device_map}")
        model_loading_args["use_flash_attn"] = use_flash_attn

    # Load tokenizer first
    tokenizer_config = {"trust_remote_code": trust_remote_code}

    if local_files_only:
        tokenizer_config["local_files_only"] = True

    # Suppress tokenizer warnings before loading
    suppress_tokenizer_warnings()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)
        logger.info("Tokenizer loaded successfully")
    except (HFValidationError, Exception) as e:
        if isinstance(e, HFValidationError):
            logger.error(f"HuggingFace validation error - forcing local-only mode: {e}")
        else:
            logger.error(f"Error loading tokenizer: {e}")
        
        logger.info("Trying to load tokenizer manually using Qwen tokenizer...")
        # For InternVL models, use the underlying Qwen tokenizer directly
        try:
            from transformers import Qwen2TokenizerFast
            logger.info("Using Qwen2TokenizerFast directly...")
            tokenizer = Qwen2TokenizerFast.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception as qwen_error:
            logger.error(f"Error loading Qwen tokenizer: {qwen_error}")
            logger.info("Trying with basic AutoTokenizer and local files only...")
            # Last resort - force local files only
            minimal_config = {
                "trust_remote_code": True,
                "local_files_only": True
            }
            tokenizer = AutoTokenizer.from_pretrained(model_path, **minimal_config)

    # Load model with warnings suppressed
    try:
        suppress_tokenizer_warnings()  # Suppress warnings again before model loading
        model = AutoModel.from_pretrained(model_path, **model_loading_args).eval()

        # Move model to appropriate device (only for CPU and single GPU)
        if (
            device_type == "cuda"
            and num_gpus == 1
            and "device_map" not in model_loading_args
        ):
            model = model.cuda()
            logger.info("Model loaded on GPU")
        elif device_type == "cpu":
            logger.info("Model loaded on CPU")
        else:
            logger.info(f"Model loaded with device mapping across {num_gpus} GPUs")

        logger.info(f"Model loaded successfully on {device_type}!")
        return model, tokenizer

    except (HFValidationError, Exception) as e:
        if isinstance(e, HFValidationError):
            logger.error(f"HuggingFace validation error during model loading - forcing local-only mode: {e}")
        else:
            logger.error(f"Error loading model: {e}")
        logger.info("Trying fallback loading...")

        # Fallback loading with local-only and simplified arguments
        fallback_args = {
            "trust_remote_code": True,
            "local_files_only": True  # Force local files only
        }

        # Apply basic device configuration for fallback
        if device_type == "cpu":
            fallback_args["torch_dtype"] = torch.float32
        else:
            fallback_args["torch_dtype"] = torch.bfloat16

        suppress_tokenizer_warnings()  # Suppress warnings for fallback loading too
        model = AutoModel.from_pretrained(model_path, **fallback_args).eval()

        # Move to device for single GPU or CPU
        if device_type == "cuda" and num_gpus == 1:
            model = model.cuda()
        elif device_type == "cpu":
            pass  # Already on CPU

        logger.info("Model loaded with fallback parameters")
        return model, tokenizer
