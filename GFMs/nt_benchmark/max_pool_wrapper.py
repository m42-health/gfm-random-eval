"""Creates the model that uses max pooling for feature aggregation."""

import torch
import torch.nn as nn


class MaxPoolWrapper(nn.Module):
    """Wrapper class that uses max pooling for feature aggregation."""

    def __init__(
        self: "MaxPoolWrapper", base_model: nn.Module, model_type: str, num_labels: int
    ) -> None:
        """Initialize the MaxPoolWrapper.

        Args:
            base_model (nn.Module): The base model to wrap.
            model_type (str): The model type (HyenaDNA, Caduceus, DNABERTv2, etc.).
            num_labels (int): The number of labels for a classification task.

        """
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type.lower()
        if "hyena" in self.model_type:
            base_model.config.hidden_size = self.base_model.config.d_model
            print("Added hidden size to Hyena config")
        elif "caduceus" in self.model_type:
            base_model.config.hidden_size = self.base_model.config.d_model * 2
            print("Added hidden size to Caduceus config")
        self.score = nn.Linear(base_model.config.hidden_size, num_labels, bias=False)

    def forward(
        self: "MaxPoolWrapper",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Forward pass for the MaxPoolWrapper.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor | None): The attention mask.
            labels (torch.Tensor | None): The labels.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor]: The loss and logits.

        """
        if "hyena" in self.model_type or "caduceus" in self.model_type:
            output = self.base_model(input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states[-1]
        elif "nt" in self.model_type or "llama" in self.model_type or "mistral" in self.model_type:
            output = self.base_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            hidden_states = output.hidden_states[-1]
        elif "dnabert" in self.model_type:
            hidden_states = self.base_model(input_ids, output_hidden_states=True)[1]
        elif "genalm" in self.model_type:
            hidden_states = self.base_model(input_ids, attention_mask=attention_mask).hidden_states[
                -1
            ]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        max_pooled = hidden_states.max(dim=1)[0]
        logits = self.score(max_pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.score.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else logits
