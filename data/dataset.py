import torch 
from src.model_rf import process_dataset
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd
from typing import Callable, Optional
from utils.utils import row_to_json
from utils.create_vignette import generate_patient_data_vignette
from utils.prompts import TASK_SPECIFIC_INSTRUCTIONS

class WrapperDataset(Dataset):
    """Wraps X (DataFrame) and y (Series) and builds JSON prompt + answer/meta."""
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        answer = self.y.iloc[idx]
        prompt = row_to_json(row.drop(labels=["record_id"]))
        # return (prompt, [answer, record_id])
        return prompt, [int(answer), row["record_id"]]

class BalancedBatchSampler(Sampler):
    """Yields index lists with 50% positives / 50% negatives (replacement if needed)."""
    def __init__(self, pos_idx, neg_idx, batch_size: int, drop_last: bool = True, seed: Optional[int] = None):
        assert batch_size % 2 == 0, "batch_size must be even."
        self.pos_idx = np.asarray(pos_idx, dtype=np.int64)
        self.neg_idx = np.asarray(neg_idx, dtype=np.int64)
        self.half = batch_size // 2
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.pos_idx)
        rng.shuffle(self.neg_idx)

        def take(arr, start, n):
            if start + n <= len(arr):
                return arr[start:start+n]
            remaining = max(0, len(arr) - start)
            head = arr[start:start+remaining] if remaining > 0 else np.empty(0, dtype=np.int64)
            need = n - remaining
            tail = rng.choice(arr, size=need, replace=True) if need > 0 and len(arr) else np.empty(0, dtype=np.int64)
            return np.concatenate([head, tail])

        steps = (max(len(self.pos_idx), len(self.neg_idx)) + self.half - 1) // self.half
        for b in range(steps):
            p = take(self.pos_idx, b*self.half, self.half)
            n = take(self.neg_idx, b*self.half, self.half)
            if (len(p) < self.half or len(n) < self.half) and self.drop_last:
                break
            batch = np.concatenate([p, n])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return (max(len(self.pos_idx), len(self.neg_idx)) + self.half - 1) // self.half

def default_collate(batch):
    """Batch is a list of (prompt, [label, record_id]). Return lists."""
    prompts = [b[0] for b in batch]
    answers = [b[1] for b in batch]
    return prompts, answers

class InferenceDataset(Dataset):
    """
    Lightning-style DataModule (without inheritance) to organize loaders.
    Provides:
      - train_dataloader()
      - val_dataloader()
      - test_dataloader()
      - balanced_eval_dataloader() for LLM evaluation
    """
    def __init__(
        self,
        train_test_data_folder: str,
        batch_size_train: int = 16,
        batch_size_eval: int = 8,
        num_workers: int = 0,
        tokenizer: Callable = None,
    ):
        self.X_train, self.X_train_raw, self.X_test, self.X_test_raw, self.y_train, self.y_test = process_dataset(train_test_data_folder,include = True)
        self.batch_size_eval = batch_size_eval
        self.batch_size_train = batch_size_train
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        self.ds_train = WrapperDataset(self.X_train, self.y_train)
        self.ds_test = WrapperDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        # build balanced batch sampler (50/50) for training
        y_np = self.y_train.astype(int).to_numpy()
        pos_idx = np.where(y_np == 1)[0]
        neg_idx = np.where(y_np == 0)[0]
        batch_sampler = BalancedBatchSampler(
            pos_idx,
            neg_idx,
            batch_size=self.batch_size_train,
            drop_last=True
        )
        
        return DataLoader(
            self.ds_train,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def collate_fn(self, batch):
        prompts, answers = [], []
        for question, answer in batch:        
            question = question.replace('Patient data:', '').strip().replace('null', 'None')
            question = eval(question)
            question = generate_patient_data_vignette(question)

            chat_prompt = [
                {'role': 'system', 'content': TASK_SPECIFIC_INSTRUCTIONS},
                {'role': 'user', 'content': question}
            ]

            prompt = self.tokenizer.apply_chat_template(
                chat_prompt,
                tokenize=False,
                add_generation_prompt=True  # important so it appends the right model tag
            )

            prompts.append(prompt)
            answers.append(answer)

        return prompts, answers

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size_train,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

