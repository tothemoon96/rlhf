# coding=utf-8
from dataclasses import dataclass, field
from functools import partial
import math
import os
import sys
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
    set_seed,
    Trainer
)
from transformers.utils import PaddingStrategy, add_start_docstrings
from transformers.trainer_utils import get_last_checkpoint, IntervalStrategy
from torch import nn

from trl import RewardConfig
import logging
from multiprocessing import cpu_count

tqdm.pandas()
logger = logging.getLogger(__name__)


def print_rank_0(msg, log_file, rank):
    if rank in [-1, 0]:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")


@dataclass
@add_start_docstrings(RewardConfig.__doc__)
class TrainingArguments(RewardConfig):
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "training batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "evaluating batch size"}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of training epochs"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    report_to: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    save_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    max_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    bf16: Optional[bool] = field(default=True, metadata={"help": "bfloat16"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "float16"})
    weight_decay: float = field(
        default=0.001, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    eval_steps: Optional[float] = field(
        default=500,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_steps: Optional[float] = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    ddp_timeout: Optional[int] = field(
        default=3600,
        metadata={
            "help": "Overrides the default timeout for distributed training (value should be given in seconds)."
        },
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    warmup_steps: Optional[int] = field(
        default=1000, metadata={"help": "Linear warmup over warmup_steps."}
    )
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of update steps between two logs"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    dataloader_drop_last: Optional[bool] = field(
        default=True, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )


@dataclass
class ScriptArguments:
    """
    Hyperparameters to fine-tune a reward model on a given dataset with the `RewardTrainer`.
    """
    model_name: str = field(default=None, metadata={"help": "the model name"})
    train_data: str = field(default=None, metadata={"help": "train data path"})
    eval_data: str = field(default=None, metadata={"help": "eval data path"})
    cache_dir: str = field(default="hf_cache_dir", metadata={"help": "cache dir"})
    use_llama: Optional[bool] = field(default=True, metadata={"help": "bfloat16"})


# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(tokenizer: PreTrainedTokenizerBase, examples: Dict[str, Any]):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, add_special_tokens=False)
        tokenized_rejected = tokenizer(rejected, add_special_tokens=False)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(
            tokenized_rejected["attention_mask"]
        )

    return new_examples


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        )[0]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        )[0]
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

def main():
    parser = HfArgumentParser((TrainingArguments, ScriptArguments))
    training_args, script_args = parser.parse_args_into_dataclasses()
    log_file = os.path.join(training_args.output_dir, "print_log.txt")
    rank = training_args.distributed_state.process_index

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, fp16-bits training: {training_args.fp16}, bf16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    training_args._frozen = False
    training_args.data_seed = training_args.seed

    # Load the dataset and pre-process it
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if script_args.use_llama:
        tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<unk>",
            }
        )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    with training_args.distributed_state.main_process_first():
        train_dataset = load_dataset(
            "json", data_files=script_args.train_data, cache_dir=script_args.cache_dir
        )["train"]
        eval_dataset = load_dataset(
            "json", data_files=script_args.eval_data, cache_dir=script_args.cache_dir
        )["train"]

        # Preprocess the dataset and filter out examples that are longer than training_args.max_length
        train_dataset = train_dataset.map(
            partial(preprocess_function, tokenizer),
            batched=True,
            num_proc=max(cpu_count() // 2, 1),
            remove_columns=["chosen", "rejected"],
        )
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= training_args.max_length
            and len(x["input_ids_rejected"]) <= training_args.max_length
        )

        eval_dataset = eval_dataset.map(
            partial(preprocess_function, tokenizer),
            batched=True,
            num_proc=max(cpu_count() // 2, 1),
            remove_columns=["chosen", "rejected"],
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= training_args.max_length
            and len(x["input_ids_rejected"]) <= training_args.max_length
        )

    for i in range(2):
        print_rank_0("Eval tokenized example: {}".format(train_dataset[i]), log_file, rank)
    for i in range(2):
        print_rank_0("Train tokenized example: {}".format(eval_dataset[i]), log_file, rank)

    # Define the training arguments
    training_nums = len(train_dataset)
    global_batch_size = (
        training_args.distributed_state.num_processes
        * training_args.gradient_accumulation_steps
        * training_args.per_device_train_batch_size
    )
    if training_args.dataloader_drop_last:
        num_steps = (
            math.floor(training_nums / global_batch_size) * training_args.num_train_epochs
        )
    else:
        num_steps = (
            math.ceil(training_nums / global_batch_size) * training_args.num_train_epochs
        )
    eval_steps = max(num_steps // (training_args.num_train_epochs * 4), 5)
    print_rank_0(
        "num_gpus = {}, training_nums = {}, num_steps = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            training_args.distributed_state.num_processes,
            training_nums,
            num_steps,
            training_args.warmup_steps,
            eval_steps,
            eval_steps,
        ),
        log_file,
        rank
    )
    training_args.eval_steps = eval_steps
    training_args.save_steps = eval_steps

    # Model must be loaded after create `TrainingArguments`!!!
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
    )
    model.config.pad_token_id = 0

    # Define the Trainer
    model.config.use_cache = False
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8
        )
    )

    # Training
    trainer.train()
    # Saving
    trainer.save_model(training_args.output_dir)
    print_rank_0(
        "\n Training completed!!! If there's a warning about missing keys above, please disregard :)",
        log_file,
        rank
    )


if __name__ == "__main__":
    main()
