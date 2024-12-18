from tokenizers_impl import CharacterTokenizerLlama
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    MistralModel,
)


def load_model_and_tokenizer(
    model_name: str, pretrained: bool
) -> tuple[AutoModel, AutoTokenizer, int]:
    """Load a model and tokenizer based on the model name.

    Args:
        model_name (str): The name of the model to load.
        pretrained (bool): Whether to load the pretrained model.

    Returns:
        tuple[AutoModel, AutoTokenizer, int]: A tuple containing the model, tokenizer, and max length.

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
        model, tokenizer, max_length = load_function(pretrained)
    else:
        raise NotImplementedError(f"Unsupported model name: {model_name}")

    print(f"Loaded model and tokenizer based on {model_name}")
    # print(f"Model: {model}")
    # print(f"Tokenizer: {tokenizer}")
    print(f"Max length: {max_length}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    model.cuda()
    model.eval()
    return model, tokenizer, max_length


def load_nt_500m(pretrained: bool) -> tuple[AutoModel, AutoTokenizer, int]:  # noqa
    model_name = "InstaDeepAI/nucleotide-transformer-500m-1000g"
    if pretrained:
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = 1000
    return model, tokenizer, max_length


def load_nt_50m(pretrained: bool) -> tuple[AutoModel, AutoTokenizer, int]:  # noqa
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    if pretrained:
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    max_length = 2048
    return model, tokenizer, max_length


def load_dnabertv2(pretrained: bool) -> tuple[AutoModel, AutoTokenizer, int]:  # noqa
    model_name = "zhihan1996/DNABERT-2-117M"
    if pretrained:
        config = BertConfig.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True, config=config
        )
    else:
        config = BertConfig.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    max_length = 512
    return model, tokenizer, max_length


def load_hyenadna(pretrained: bool) -> tuple[AutoModel, AutoTokenizer, int]:  # noqa
    model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    max_length = 1024
    return model, tokenizer, max_length


def load_mistral(pretrained: bool) -> tuple[AutoModel, AutoTokenizer, int]:  # noqa
    model_path = "/models_gfm/variant_paper/mistral/weighed-mistral-500m-w100-all-hg1000-character-4096-train_small-shift-50-rc-1e-in9s5m4k"
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_config.use_cache = False
    if pretrained:
        model = MistralModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            config=model_config,
        )
    else:
        model = MistralModel(config=model_config)
    tokenizer = CharacterTokenizerLlama(characters=list("DNATGC"), model_max_length=None)
    tokenizer.pad_token_id = 9
    tokenizer.eos_token_id = 10
    tokenizer.bos_token_id = 11

    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Using char tokenizer with len(vocab) = {len(tokenizer.get_vocab())}")
    max_length = 4096
    return model, tokenizer, max_length


def load_genalm(pretrained: bool) -> tuple[AutoModel, AutoTokenizer, int]:  # noqa
    model_name = "AIRI-Institute/gena-lm-bert-base-t2t"
    if pretrained:
        model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, trust_remote_code=True
        )
    else:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, output_hidden_states=True
        )
        model = AutoModel.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    max_length = 512
    return model, tokenizer, max_length


def load_caduceus(pretrained: bool) -> tuple[AutoModel, AutoTokenizer, int]:  # noqa
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if pretrained:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )
    else:
        model = AutoModelForMaskedLM.from_config(model_config, trust_remote_code=True)
    max_length = 131072
    return model, tokenizer, max_length
