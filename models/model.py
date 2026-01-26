import ast
import json
import os
from utils import get_header
import time
import numpy as np
import pandas as pd
from tasks.task import Task
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
from typing import Dict
from PIL import Image
warnings.filterwarnings('ignore')


class Model():
    """
    Base Model class that other models will inherit from.
    """
    def __init__(
            self,
            task: Task
    ):
        self.task = task

    def run(self):
        print('Need to specify a particular model class.')
        raise NotImplementedError
    
    def save_results(self, results_file: str=None):
        """
        Save the task results to a CSV file.
        """
        if results_file:
            self.task.results_df.to_csv(results_file, index=False)
        else:
            filename = f'results_{time.time()}.csv'
            self.task.results_df.to_csv(filename, index=False)

class ParseModel(Model):
    """
    Model class responsible for parsing responses from an LLM.
    """
    def __init__(
            self,
            model_name: str,
            payload_path: str,
            api_file: str,
            sleep: int = 0,
            prompt_path: str = None
    ):
        ### [CHECK] Shouldn't we add this: super().__init__(task=None) ?
        self.model_name = model_name
        ### [CHECK] Wanna keep the name good_models?
        good_models = ['gpt4']
        assert self.model_name in good_models, (
            f"Model name must be one of {good_models}, not {self.model_name}"
        )

        with open(payload_path) as f:
            self.payload = json.load(f)
        with open(api_file, 'r') as f:
            self.api_metadata = json.load(f)

        self.prompt = open(prompt_path, 'r').read() if prompt_path else ""
        self.header = get_header(self.api_metadata, model=self.model_name)
        self.api_metadata = self.api_metadata[self.model_name]
        self.sleep = sleep


class APIModel(Model):
    """
    Model class for interacting with various APIs.
    """
    def __init__(
            self,
            task: Task,
            model_name: str,
            payload_path: str,
            api_file: str,
            sleep: int = 0,
            shuffle: bool = False,
            n_trials: int = None,
            parse_model: ParseModel = None
    ):
        self.model_name = model_name
        good_models = ['gpt4v', 'gpt4o', 'claude-sonnet', 'claude-opus', 'gemini-ultra', 'stable-diffusion', 'dalle', 'parti', 'muse']
        assert self.model_name in good_models, f'Model name must be one of {str(good_models)}, not {self.model_name}'
        super().__init__(task)
        self.payload = json.load(open(payload_path))
        self.api_metadata = json.load(open(api_file, 'r'))
        self.header = get_header(self.api_metadata, model=self.model_name)
        self.api_metadata = self.api_metadata[self.model_name]
        self.sleep = sleep
        self.results_file = self.task.results_path
        self.shuffle = shuffle
        self.n_trials = n_trials
        # Define the parse model, if necessary.
        self.parse_model = parse_model
        if self.parse_model:
            self.task.results_df['answer'] = ''
            print(f'Parse model: {self.parse_model.model_name}')
        # Shuffle and subsample the task dataset, if necessary
        if self.shuffle:
            if self.n_trials and self.n_trials > 0:
                self.task.results_df = self.task.results_df.sample(n=self.n_trials)
            else:
                self.task.results_df = self.task.results_df.sample(frac=1)

    #@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(10))
    def run(self):
        
        if self.task.num_remaining_trials() == 0:
            return
        
        p_bar = tqdm(total=self.task.num_remaining_trials())
        
        for i, trial in self.task.results_df.iterrows():
            if type(trial.response) != str or trial.response == '0.0':
                trial_payload = self.payload.copy()
                task_payload = self.build_vlm_payload(trial, trial_payload)
                response = self.run_trial(self.header, self.api_metadata, task_payload)
                self.task.results_df.loc[i, 'response'] = response

                # Parse the response, if necessary.
                if self.parse_model:
                    self.task.results_df.loc[i, 'answer'] = self.parse_model.parse_response(response)

                p_bar.update()
                time.sleep(self.sleep)
                
            if i % 1 == 0:
                self.save_results(self.results_file)
                
        self.save_results(self.results_file)


class T2IModel(APIModel):
    """
    Text-to-Image Model class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_path = os.path.join(self.task.output_dir, 't2i', self.task.task_name, self.model_name)
        os.makedirs(self.img_path, exist_ok=True)

    #@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(10))
    def run(self):
        p_bar = tqdm(total=self.task.num_remaining_trials())
        for i, trial in self.task.results_df.iterrows():
            if not bool(trial.completed):
                trial_payload = self.payload.copy()
                task_payload = self.build_payload(trial, trial_payload)
                image, revised_prompt = self.run_trial(self.header, self.api_metadata, task_payload)
                self.task.results_df.loc[i, 'revised_prompt'] = revised_prompt
                self.task.results_df.loc[i, 'completed'] = True
                path = os.path.join(self.img_path, trial.path)
                self.task.results_df.loc[i, 'path'] = path
                image.save(path)
                p_bar.update()
                time.sleep(self.sleep)
            if i % 1 == 0:
                self.save_results(self.results_file)
        self.save_results(self.results_file)

class LocalLanguageModel(Model):
    """
    Local Language Model class for huggingface models.
    """
    def __init__(
        self,
        task: Task = None,
        max_parse_tokens: int = 256,
        prompt_format: str = None,
        weights_path: str = None,
        probe_layers: Dict = None
    ):
        super().__init__(task)
        if task:
            self.results_file = self.task.results_path
        self.max_parse_tokens = max_parse_tokens
        self.prompt_format = prompt_format
        self.weights_path = weights_path
        self.probe_layers = probe_layers
        self.prompt = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.activations = {}
        self.llm = AutoModelForCausalLM.from_pretrained(self.weights_path, device_map='auto', torch_dtype='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, use_fast=True)
        self.llm.eval()
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.target_column = 'response'

    def run_batch(self, batch):
        prompts = [p.format(text_to_parse=t) for p, t in zip([self.prompt]*len(batch), batch[self.target_column])]
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=self.max_parse_tokens)
        outputs = [output[inputs.input_ids.shape[1]:] for output in outputs]
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, o in zip(prompts, decoded_outputs):
            print(f'Prompt: {i}\nResponse: {o}\n')
        batch['answer'] = decoded_outputs
        return batch

class LocalVLModel(Model):
    """
    Local Vision-Language Model class for processing multimodal tasks.
    """
    def __init__(
        self,
        task: Task = None,
        max_tokens: int = 512,
        batch_size: int = 32,
        weights_path: str = None,
        shuffle: bool = False,
        n_trials: int = None,
        model_name: str = None
    ):
        super().__init__(task)
        self.results_file = self.task.results_path if self.task is not None else None
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.weights_path = weights_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_trials = n_trials
        self.shuffle = shuffle
        self.model_name = model_name

        # shuffle and subsample the task dataset, if necessary
        if self.shuffle and self.task is not None:
            if self.n_trials and self.n_trials > 0:
                self.task.results_df = self.task.results_df.sample(n=self.n_trials)
            else:
                self.task.results_df = self.task.results_df.sample(frac=1)

    def _select_image_column(self, batch: pd.DataFrame) -> str:
        """Return the column name containing image paths for the current task."""
        candidate_columns = ['path', 'unified_path']
        for column in candidate_columns:
            if column in batch.columns:
                return column
        fallback_columns = [
            column for column in batch.columns
            if isinstance(column, str) and 'path' in column.lower() and 'decomposed' not in column.lower()
        ]
        if fallback_columns:
            return fallback_columns[0]
        raise KeyError(
            'Unable to find an image path column in batch. '
            f'Available columns: {list(batch.columns)}'
        )

    def _normalize_image_path(self, image_path: str) -> str:
        """Convert relative paths to absolute ones anchored at the task root."""
        if not isinstance(image_path, str):
            return image_path
        image_path = image_path.strip()
        if os.path.isabs(image_path):
            return image_path
        if getattr(self.task, 'root_dir', None):
            candidate = os.path.join(self.task.root_dir, image_path)
            if os.path.exists(candidate):
                return candidate
        return image_path

    def _is_missing(self, value):
        return value is None or (isinstance(value, float) and np.isnan(value))

    def get_image_paths(self, batch: pd.DataFrame):
        """Resolve the correct set of image paths for this batch."""
        column = self._select_image_column(batch)
        raw_paths = batch[column].tolist()
        return [self._normalize_image_path(path) for path in raw_paths]

    def get_decomposed_paths(self, row: pd.Series):
        """Return normalized decomposed image paths for a trial, if available."""
        if not isinstance(row, pd.Series) or 'decomposed_paths' not in row:
            return []
        value = row['decomposed_paths']
        if self._is_missing(value):
            return []
        if isinstance(value, str):
            try:
                paths = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                paths = []
        elif isinstance(value, (list, tuple, np.ndarray)):
            paths = list(value)
        else:
            paths = []
        normalized = [
            self._normalize_image_path(path)
            for path in paths
            if isinstance(path, str)
        ]
        return normalized

    def format_prompt(self, row: pd.Series) -> str:
        """Format task prompt with trial-specific metadata when applicable."""
        prompt = self.task.prompt
        if getattr(self.task, 'task_name', None) != 'rmts':
            return prompt
        replacements = {}
        for key in ['feature', 'object_loc', 'object_ind', 'pair', 'pair_loc', 'relation']:
            if key in row and not self._is_missing(row[key]):
                replacements[key] = row[key]
        if not replacements:
            return prompt
        try:
            return prompt.format(**replacements)
        except KeyError:
            return prompt

    def get_trial_images(self, row: pd.Series, default_path: str):
        """Return the list of PIL.Images to feed to the VLM for this trial."""
        default_path = self._normalize_image_path(default_path)
        condition = getattr(self.task, 'condition', None)
        if condition == 'decomposed':
            decomposed_paths = self.get_decomposed_paths(row)
            images = []
            for path in decomposed_paths:
                try:
                    images.append(Image.open(path).convert('RGB'))
                except (FileNotFoundError, OSError):
                    continue
            if images:
                return images
        return [Image.open(default_path).convert('RGB')]

    def compose_images(self, images):
        """Combine multiple images horizontally for models that need a single canvas."""
        if not images:
            raise ValueError('No images provided to compose.')
        if len(images) == 1:
            return images[0]
        width = sum(img.width for img in images)
        height = max(img.height for img in images)
        canvas = Image.new('RGB', (width, height), (255, 255, 255))
        offset = 0
        for img in images:
            canvas.paste(img, (offset, 0))
            offset += img.width
        return canvas

    def run(self):
        if 'response' not in self.task.results_df.columns:
            self.task.results_df['response'] = np.nan

        remaining = self.task.results_df[self.task.results_df['response'].isna()]
        if remaining.empty:
            return

        batches = np.array_split(
            remaining,
            np.ceil(len(remaining) / self.batch_size)
        )
        for i, batch in tqdm(
            enumerate(batches),
            total=len(batches)
        ):
            processed = self.run_batch(batch)
            # Write processed columns back into the main results DataFrame.
            for col in processed.columns:
                if col in self.task.results_df.columns:
                    self.task.results_df.loc[processed.index, col] = processed[col]
            if i % 10 == 0:
                self.save_results(self.results_file)
        self.save_results(self.results_file)
