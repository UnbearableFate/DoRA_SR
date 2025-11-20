
import torch


class DataCollator:
    def __init__(self, eos_token_id, max_length=None):
        self.eos_token_id = eos_token_id
        self.max_length = max_length

    def __call__(self, batch):
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        input_lengths = torch.stack(batch["input_length"])
        prompt_lengths = torch.stack(batch["prompt_length"])
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"], batch_first=True, padding_value=self.eos_token_id
        )
        col_indices = torch.arange(input_ids.size(1)).unsqueeze(0)
        attention_mask = col_indices < input_lengths.unsqueeze(1)
        label_mask = torch.logical_or(col_indices < prompt_lengths.unsqueeze(1), ~attention_mask)
        labels = input_ids.masked_fill(label_mask, -100)
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            labels = labels[:, : self.max_length]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}