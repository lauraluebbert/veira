import torch 
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd
import pickle
from typing import Callable, Optional
from utils.utils import row_to_json
from utils.create_vignette import generate_patient_data_vignette
from utils.prompts import TASK_SPECIFIC_INSTRUCTIONS
from src.utils import data_to_use
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import json
import re

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
        data_path: str,
        col_meta_path: str,
        data_dictionary_path:str, 
        batch_size_train: int = 16,
        batch_size_eval: int = 8,
        num_workers: int = 0,
        tokenizer: Callable = None,
    ):
        self.data_df = pd.read_csv(data_path)
        self.col_meta = pd.read_excel(col_meta_path)
        self.data_dict = pd.read_csv(data_dictionary_path)
        self.generate_field_definitions()

        self.X_train, self.X_train_raw, self.X_test, self.X_test_raw, self.y_train, self.y_test = self.process_dataset(train_test_data_folder,include = True)
        self.batch_size_eval = batch_size_eval
        self.batch_size_train = batch_size_train
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        self.ds_train = WrapperDataset(self.X_train_raw, self.y_train)
        self.ds_test = WrapperDataset(self.X_test_raw, self.y_test)

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
            question = generate_patient_data_vignette(question, self.df_field_definitions)

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
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def generate_field_definitions(self):
        self.filtered_fields = self.data_dict[
            self.data_dict["Variable / Field Name"].apply(
                lambda x: any(str(x) in col for col in self.data_df.columns))
        ][[
            "Variable / Field Name",
            "Field Label",
            "Choices, Calculations, OR Slider Labels",
            "Field Note"
        ]]
        self.field_definitions_list = self.filtered_fields.to_dict(orient="records")
        self.cleaned_field_definitions = [
            self.clean_field_dict(d) for d in self.field_definitions_list]
        self.field_definitions = json.dumps(self.cleaned_field_definitions)
        self.df_field_definitions = pd.DataFrame(eval(self.field_definitions))
        self.df_field_definitions = self.df_field_definitions.rename(columns={
            "Variable / Field Name": "Field Name",
            "Choices, Calculations, OR Slider Labels": "Choices"
        })

    def process_dataset(self, train_test_data_folder, include = False):
        pathogen = "all-viral"
        num_cols_present, cat_cols_present = self.get_standard_features()

        with open(f"{train_test_data_folder}/X_train_{pathogen}.pkl", "rb") as f:
            X_train_raw = pickle.load(f)
            # Drop record_id column
            if "record_id" in X_train_raw.columns and include is False:
                X_train_raw = X_train_raw.drop(columns=["record_id"])
        with open(f"{train_test_data_folder}/X_test_{pathogen}.pkl", "rb") as f:
            X_test_raw = pickle.load(f)
            if "record_id" in X_test_raw.columns and include is False:
                X_test_raw = X_test_raw.drop(columns=["record_id"])
        with open(f"{train_test_data_folder}/y_train_{pathogen}.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open(f"{train_test_data_folder}/y_test_{pathogen}.pkl", "rb") as f:
            y_test = pickle.load(f)

        # Fit preprocessor on training data and process X
        local_preprocessor = self.preprocess_data(
            num_cols_present, cat_cols_present).fit(X_train_raw)
        X_train = local_preprocessor.transform(X_train_raw)
        X_test = local_preprocessor.transform(X_test_raw)

        return X_train, X_train_raw, X_test, X_test_raw, y_train, y_test

    def get_standard_features(self):
        data_df_models = data_to_use(self.data_df)

        num_cols = self.col_meta[self.col_meta["data_type"] == "numerical"]["column"].values
        num_cols_present = [
            col for col in num_cols if col in data_df_models.columns]
        cat_cols = self.col_meta[self.col_meta["data_type"]
                            == "categorical"]["column"].values
        cat_cols_present = [
            col for col in cat_cols if col in data_df_models.columns]

        return num_cols_present, cat_cols_present

    def preprocess_data(self, num_cols_present, cat_cols_present):
        # Preprocessing pipeline
        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        # One-hot encode categorical variables
        categorical_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, num_cols_present),
            ('cat', categorical_pipeline, cat_cols_present)
        ])

        return preprocessor

    def strip_html_and_whitespace(self, text):
        """
        Remove HTML tags and decode basic HTML entities, then strip whitespace.
        """
        if not isinstance(text, str):
            return text
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Replace HTML entities for non-breaking space and quotes
        text = text.replace('\\u00a0', ' ').replace('&nbsp;', ' ')
        # Remove extra whitespace
        text = text.strip()
        # Remove any remaining curly braces and their contents (e.g., {participantid_country})
        text = re.sub(r'\{.*?\}', '', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text

    def clean_field_dict(self, d):
        cleaned = {}
        for k, v in d.items():
            # Remove keys with None, 'null' (as string), or NaN
            if v is None or (isinstance(v, str) and v.strip().lower() == "null") or (isinstance(v, str) and v.strip().lower() == "nan") or pd.isna(v):
                continue
            # Shorten key names as specified
            if k == "Variable / Field Name":
                new_key = "Field Name"
            elif k == "Choices, Calculations, OR Slider Labels":
                new_key = "Choices"
            else:
                new_key = k
            # Remove unnecessary backslashes
            if isinstance(v, str):
                v = v.replace("\\/", "/").replace('\\"', '"')
            # For "Field Label" and "Field Note", strip HTML and whitespace
            if new_key in ["Field Label", "Field Note"]:
                v = self.strip_html_and_whitespace(v)
            cleaned[new_key] = v
        return cleaned





