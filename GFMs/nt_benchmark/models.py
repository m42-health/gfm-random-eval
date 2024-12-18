"""Load models and tokenizers."""

import torch.nn as nn
import transformers
from config import Config
from ft_datasets import nt_benchmarks
from max_pool_wrapper import MaxPoolWrapper
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from character_tokenizer import CharacterTokenizer

transformers.utils.TRUST_REMOTE_CODE = True


def load_model_tokenizer(config: Config) -> tuple[nn.Module, PreTrainedTokenizer]:
    """Load a model and tokenizer for a given model type and weight init (pretrained or random).

    Args:
        config (Config): The configuration for the NT benchmark.

    Returns:
        tuple[nn.Module, PreTrainedTokenizer]: The model and tokenizer.

    """
    model_types = {
        "nt_50m_max_pool": load_nt_50m_max_pool,
        "nt_500m_max_pool": load_nt_500m_max_pool,
        "hyenadna_max_pool": load_hyena_max_pool,
        "genalm_max_pool": load_genalm_max_pool,
        "dnabert_max_pool": load_dnabert_max_pool,
        "caduceus_max_pool": load_caduceus_max_pool,
        "mistral_max_pool": load_mistral_max_pool,
    }

    load_function = model_types.get(config.model_type)
    if load_function:
        model, tokenizer = load_function(config)
    else:
        raise NotImplementedError("Model type must be one of: " + ", ".join(model_types.keys()))

    print(f"Loaded model and tokenizer based on {config.model_type}")
    print(model)
    print(tokenizer)
    return model, tokenizer


def load_caduceus_max_pool(config: Config) -> tuple[nn.Module, PreTrainedTokenizer]:  # noqa
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
    model_config.fused_add_norm = False
    if config.weight_init == "random":
        base_model = AutoModelForMaskedLM.from_config(model_config, trust_remote_code=True)
        print("Initializing random Caduceus model with max pooling")
    elif config.weight_init == "pretrained":
        base_model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )
        print("Using pre-trained Caduceus model with max pooling")
    else:
        raise ValueError("Invalid weight initialization type. Please use 'random' or 'pretrained'.")
    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_hyena_max_pool(config: Config) -> tuple[nn.Module, PreTrainedTokenizer]:  # noqa
    model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.weight_init == "random":
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        base_model = AutoModel.from_config(model_config, trust_remote_code=True)
        print("Initializing random Hyena model")
    elif config.weight_init == "pretrained":
        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print("Using pre-trained Hyena model")
    else:
        raise ValueError("Invalid weight initialization type. Please use 'random' or 'pretrained'.")
    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_genalm_max_pool(config: Config) -> tuple[nn.Module, PreTrainedTokenizer]:  # noqa
    model_name = "AIRI-Institute/gena-lm-bert-base-t2t"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.weight_init == "random":
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
        model_config.output_hidden_states = True
        base_model = AutoModel.from_config(model_config)
        print("Initializing random GenaLM model")
    elif config.weight_init == "pretrained":
        base_model = AutoModel.from_pretrained(
            model_name,
            num_labels=config.num_classes,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        print("Using pre-trained GenaLM model")
    else:
        raise ValueError("Invalid weight initialization type. Please use 'random' or 'pretrained'.")
    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_nt_50m_max_pool(config: Config) -> tuple[nn.Module, PreTrainedTokenizer]:  # noqa
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.weight_init == "random":
        base_model = AutoModelForMaskedLM.from_config(
            AutoConfig.from_pretrained(model_name, trust_remote_code=True),
            trust_remote_code=True,
        )
        print("Initializing random Nucleotide Transformer 50M model with max pooling")
    elif config.weight_init == "pretrained":
        base_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        print("Using pre-trained Nucleotide Transformer 50M model with max pooling")
    else:
        raise ValueError("Invalid weight initialization type. Please use 'random' or 'pretrained'.")

    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_nt_500m_max_pool(config: Config) -> tuple[nn.Module, PreTrainedTokenizer]:  # noqa
    model_name = "InstaDeepAI/nucleotide-transformer-500m-1000g"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.weight_init == "random":
        base_model = AutoModelForMaskedLM.from_config(
            AutoConfig.from_pretrained(model_name, trust_remote_code=True),
            trust_remote_code=True,
        )
        print("Initializing random Nucleotide Transformer 500M model with max pooling")
    elif config.weight_init == "pretrained":
        base_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        print("Using pre-trained Nucleotide Transformer 500M model with max pooling")
    else:
        raise ValueError("Invalid weight initialization type. Please use 'random' or 'pretrained'.")

    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_dnabert_max_pool(config: Config) -> tuple[nn.Module, PreTrainedTokenizer]:  # noqa
    model_name = "zhihan1996/DNABERT-2-117M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.weight_init == "random":
        model_config = BertConfig.from_pretrained(model_name)
        base_model = AutoModelForMaskedLM.from_config(model_config, trust_remote_code=True)
        print("Initializing random DNABERT model")
    elif config.weight_init == "pretrained":
        model_config = BertConfig.from_pretrained(model_name)
        base_model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )
        print("Using pre-trained DNABERT model")
    else:
        raise ValueError("Invalid weight initialization type. Please use 'random' or 'pretrained'.")

    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_mistral_max_pool(config: Config) -> tuple[nn.Module, PreTrainedTokenizer]:  # noqa
    model_path = (
        "/models_gfm/variant_paper/mistral/"
        "weighed-mistral-500m-w100-all-hg1000-character-4096-train_small-shift-50-rc-1e-in9s5m4k"
    )
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_config.use_cache = False

    if config.weight_init == "random":
        base_model = AutoModelForCausalLM.from_config(config=model_config)
        print("Initializing random mistral model with max pooling")
    elif config.weight_init == "pretrained":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, config=model_config, trust_remote_code=True, device_map="auto"
        )
        print("Using pre-trained mistral model with max pooling!")
    else:
        raise ValueError("Invalid weight initialization type. Please use 'random' or 'pretrained'.")

    tokenizer = CharacterTokenizer(characters=list("DNATGC"), model_max_length=None)
    tokenizer.pad_token_id = 9
    tokenizer.eos_token_id = 10
    tokenizer.bos_token_id = 11

    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )

    return model, tokenizer
