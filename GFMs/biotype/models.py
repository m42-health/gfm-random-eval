"""Load models and tokenizers for biotype experiments."""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    PreTrainedTokenizer,
)

from character_tokenizer import CharacterTokenizer


def load_model_and_tokenizer(
    model_name: str, pretrained: bool, tokenizer_type: str, embedding_dim: int = -1
) -> tuple[AutoModelForMaskedLM, PreTrainedTokenizer, int]:
    """Load a model and tokenizer based on the model name.

    Args:
        model_name (str): The name of the model to load.
        pretrained (bool): Whether to load the pretrained model.
        tokenizer_type (str): The type of tokenizer to use.
        embedding_dim (int, optional): The embedding dimension to use.
        When provided -1 - the model is loaded with default parameters.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        max_length: The maximum length of the tokenizer.

    """
    model_types = {
        "nt_500m": load_nt_500m,
        "nt_50m": load_nt_50m,
        "dnabertv2": load_dnabertv2,
        "hyenadna": load_hyenadna,
        "genalm": load_genalm,
        "caduceus": load_caduceus,
        "mistral": load_mistral,
    }

    load_function = model_types.get(model_name)
    if load_function:
        model, tokenizer, max_length = load_function(pretrained, tokenizer_type, embedding_dim)
    else:
        raise NotImplementedError(f"Unsupported model name: {model_name}")

    print(f"Loaded model and tokenizer based on {model_name}")
    print(f"Model: {model}")
    print(f"Tokenizer: {tokenizer}")
    print(f"Max length: {max_length}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    model.cuda()
    model.eval()
    return model, tokenizer, max_length


def load_nt_500m(  # noqa
    pretrained: bool, tokenizer_type: str, embedding_dim: int = -1
) -> tuple[AutoModelForMaskedLM, PreTrainedTokenizer, int]:
    model_name = "InstaDeepAI/nucleotide-transformer-500m-1000g"
    if pretrained:
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if embedding_dim != -1:
            config.hidden_size = embedding_dim
            config.intermediate_size = 4 * embedding_dim
            print(f"Using custom embedding dim - {embedding_dim}")
        model = AutoModelForMaskedLM.from_config(config)

    if tokenizer_type == "char":
        tokenizer = CharacterTokenizer(
            characters=list("DNATGC"),
            model_max_length=1000,
            padding_side="left",
        )
        config.vocab_size = len(tokenizer.get_vocab())
        set_char_embeddings_for_model(model, tokenizer)
        print(f"Using char tokenizer with len(vocab) = {len(tokenizer.get_vocab())}")
    elif tokenizer_type == "default":
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = 1000
    return model, tokenizer, max_length


def load_nt_50m(  # noqa
    pretrained: bool, tokenizer_type: str, embedding_dim: int = -1
) -> tuple[AutoModelForMaskedLM, PreTrainedTokenizer, int]:
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    if pretrained:
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if embedding_dim != -1:
            config.hidden_size = embedding_dim
            config.intermediate_size = 4 * embedding_dim
            print(f"Using custom embedding dim - {embedding_dim}")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    if tokenizer_type == "char":
        tokenizer = CharacterTokenizer(
            characters=list("DNATGC"),
            model_max_length=2048,
            padding_side="left",
        )
        config.vocab_size = len(tokenizer.get_vocab())
        set_char_embeddings_for_model(model, tokenizer)
        print(f"Using char tokenizer with len(vocab) = {len(tokenizer.get_vocab())}")
    elif tokenizer_type == "default":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    max_length = 2048
    return model, tokenizer, max_length


def load_dnabertv2(  # noqa
    pretrained: bool, tokenizer_type: str, embedding_dim: int = -1
) -> tuple[AutoModelForMaskedLM, PreTrainedTokenizer, int]:
    model_name = "zhihan1996/DNABERT-2-117M"
    if pretrained:
        config = BertConfig.from_pretrained(model_name)
        config.use_triton = False
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True, config=config
        )
    else:
        config = BertConfig.from_pretrained(model_name)
        config.use_triton = False
        if embedding_dim != -1:
            config.hidden_size = embedding_dim
            config.intermediate_size = 4 * embedding_dim
            print(f"Using custom embedding dim - {embedding_dim}")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    if tokenizer_type == "char":
        tokenizer = CharacterTokenizer(
            characters=list("DNATGC"),
            model_max_length=2048,
            padding_side="left",
        )
        config.vocab_size = len(tokenizer.get_vocab())
        set_char_embeddings_for_model(model, tokenizer)
        print("Using char tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    max_length = 512
    return model, tokenizer, max_length


def load_hyenadna(  # noqa
    pretrained: bool,
    tokenizer_type: str,  # noqa
    embedding_dim: int = -1,  # noqa
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer, int]:
    model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if embedding_dim != -1:
            config.d_model = embedding_dim
            config.d_inner = 4 * embedding_dim
            print(f"Using custom embedding dim - {embedding_dim}")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    max_length = 1024
    return model, tokenizer, max_length


def load_genalm(  # noqa
    pretrained: bool, tokenizer_type: str, embedding_dim: int = -1
) -> tuple[AutoModel, PreTrainedTokenizer, int]:
    model_name = "AIRI-Institute/gena-lm-bert-base-t2t"
    if pretrained:
        model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, trust_remote_code=True
        )
    else:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, output_hidden_states=True
        )
        if embedding_dim != -1:
            config.hidden_size = embedding_dim
            config.intermediate_size = 4 * embedding_dim
            print(f"Using custom embedding dim - {embedding_dim}")
        model = AutoModel.from_config(config)

    if tokenizer_type == "char":
        tokenizer = CharacterTokenizer(
            characters=list("DNATGC"),
            model_max_length=512,
            padding_side="left",
        )
        config = model.config
        config.vocab_size = len(tokenizer.get_vocab())
        set_char_embeddings_for_model(model, tokenizer)
        print(f"Using char tokenizer with len(vocab) = {len(tokenizer.get_vocab())} for GenALM")
    elif tokenizer_type == "default":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    max_length = 512
    return model, tokenizer, max_length


def load_caduceus(  # noqa
    pretrained: bool,
    tokenizer_type: str,  # noqa
    embedding_dim: int = -1,
) -> tuple[AutoModelForMaskedLM, PreTrainedTokenizer, int]:
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_config.fused_add_norm = False
    if embedding_dim != -1:
        model_config.d_model = embedding_dim
        model_config.d_inner = 4 * embedding_dim
        print(f"Using custom embedding dim - {embedding_dim}")

    if pretrained:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )
    else:
        model = AutoModelForMaskedLM.from_config(model_config, trust_remote_code=True)
    max_length = 131072
    return model, tokenizer, max_length


def load_mistral(  # noqa
    pretrained: bool,
    tokenizer_type: str,  # noqa
    embedding_dim: int = -1,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer, int]:
    model_path = (
        "/home/models_gfm/variant_paper/mistral/"
        "weighed-mistral-500m-w100-all-hg1000-character-4096-train_small-shift-50-rc-1e-in9s5m4k"
    )
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_config.use_cache = False
    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            config=model_config,
        )
    else:
        if embedding_dim != -1:
            model_config.hidden_size = embedding_dim
            model_config.intermediate_size = 4 * embedding_dim
            print(f"Using custom embedding dim - {embedding_dim}")
        model = AutoModelForCausalLM.from_config(config=model_config)
    tokenizer = CharacterTokenizer(characters=list("DNATGC"), model_max_length=None)
    print(f"Using char tokenizer with len(vocab) = {len(tokenizer.get_vocab())}")
    max_length = 4096
    return model, tokenizer, max_length


def set_char_embeddings_for_model(model: AutoModel, tokenizer: PreTrainedTokenizer) -> None:
    """Ajust the embeddings of the model to be suitable to work with char tokenizer.

    Old embeddings are replaced with randomly initialized ones.

    Args:
        model (AutoModel): The model to update the embeddings layer.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.

    """
    if "esm" in dir(model):
        old_word_embeddings = model.esm.embeddings.word_embeddings
        embedding_dim = old_word_embeddings.embedding_dim
        new_vocab_size = tokenizer.vocab_size
        new_word_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        embedding_init_std = 0.02
        with torch.no_grad():
            new_word_embeddings.weight.data.normal_(mean=0.0, std=embedding_init_std)
            new_word_embeddings.weight.data[1].zero_()
        model.esm.embeddings.word_embeddings = new_word_embeddings
        print(model.esm.embeddings)
        print("Embeddings for char tokenizer are setup")
    elif hasattr(model, "bert"):
        old_word_embeddings = model.bert.embeddings.word_embeddings
        embedding_dim = old_word_embeddings.embedding_dim
        new_vocab_size = len(tokenizer.get_vocab())
        new_word_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        embedding_init_std = 0.02
        with torch.no_grad():
            new_word_embeddings.weight.data.normal_(mean=0.0, std=embedding_init_std)
            new_word_embeddings.weight.data[tokenizer.pad_token_id].zero_()
        model.bert.embeddings.word_embeddings = new_word_embeddings
        print(model.bert.embeddings.word_embeddings)
        print("Embeddings for char tokenizer are setup")
    elif hasattr(model, "embeddings"):
        old_word_embeddings = model.embeddings.word_embeddings
        embedding_dim = old_word_embeddings.embedding_dim
        new_vocab_size = len(tokenizer.get_vocab())
        new_word_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        embedding_init_std = 0.02
        with torch.no_grad():
            new_word_embeddings.weight.data.normal_(mean=0.0, std=embedding_init_std)
            new_word_embeddings.weight.data[tokenizer.pad_token_id].zero_()
        model.embeddings.word_embeddings = new_word_embeddings
        print(model.embeddings.word_embeddings)
        print("Embeddings for char tokenizer are setup")
    else:
        raise AttributeError("Model structure not recognized. Unable to set character embeddings.")
