# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib
import shutil

import syne_tune
import torch
import transformers
import wandb
from bo_options import lora_target_map
from peft import LoraConfig, TaskType, get_peft_model
from syne_tune import Reporter
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config


def get_optimizer_and_scheduler(model, train_dataset, config):
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    kwargs_lr_scheduler = {
        "optimizer": optimizer,
        "num_warmup_steps": config.num_warmup_steps,
        "num_training_steps": (
            (len(train_dataset) - 1) // (config.finetune_train_batch_size * config.gradient_accumulation_steps) + 1
        )
        * config.epochs,
    }
    if config.lr_scheduler_type in ("cosine", "cosine_with_warmup"):
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(**kwargs_lr_scheduler)
    elif config.lr_scheduler_type in ("linear", "linear_with_warmup"):
        lr_scheduler = transformers.get_linear_schedule_with_warmup(**kwargs_lr_scheduler)
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler


class CustomTrainer(Trainer):
    def __init__(self, *args, train_loader=None, test_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.model.config.pad_token_id)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_train_dataloader(self) -> DataLoader:
        return self.train_loader

    def get_eval_dataloader(self, _) -> DataLoader:
        return self.test_loader


def finetuning_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt-finetuning", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    # Perplexity evaluation command-line arguments
    parser.add_argument(
        "--ppl-eval-dataset",
        type=str,
        help="Dataset to evaluate perplexity.",
        choices=["wikitext2", "ptb", "c4", "alpaca", "NuminaMath-CoT"],
        default="wikitext2",
    )
    parser.add_argument(
        "--ppl-eval-nsamples",
        type=int,
        help="Number of samples of the perplexity eval dataset to load.",
        default=128,
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )

    # finetuning command-line arguments
    parser.add_argument(
        "--finetune-dataset",
        type=str,
        help="Dataset to finetune on.",
        choices=["wikitext2", "ptb", "c4", "alpaca", "NuminaMath-CoT"],
        default="wikitext2",
    )
    parser.add_argument(
        "--finetune-train-nsamples",
        type=int,
        help="Number of samples to load from the train set for finetuning.",
        default=4096,
    )
    parser.add_argument(
        "--finetune-test-nsamples",
        type=int,
        help="Number of samples to load from the test set for finetuning.",
        default=128,
    )
    parser.add_argument("--finetune-train-batch-size", type=int, default=1, help="Batch size for finetuning training.")
    parser.add_argument("--finetune-test-batch-size", type=int, default=8, help="Batch size for finetuning testing.")
    parser.add_argument(
        "--finetune-train-seqlen", type=int, default=2048, help="Sequence length for finetuning training."
    )
    parser.add_argument(
        "--finetune-test-seqlen", type=int, default=2048, help="Sequence length for finetuning testing."
    )

    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.95)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--lr-scheduler-type', type=str, default="linear")
    parser.add_argument('--num-warmup-steps', type=int, default=400)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--early-stopping-patience', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--evaluation-strategy', type=str, default="steps")
    parser.add_argument('--eval-steps', type=int, default=16)
    parser.add_argument('--save-steps', type=int, default=16)
    parser.add_argument('--save-total-limit', type=int, default=1)
    parser.add_argument('--logging-steps', type=int, default=1)

    parser.add_argument('--lora-alpha', type=float, default=32.0)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-bias', type=str, default="none")

    parser.add_argument(
        '--st_checkpoint_dir', type=str, default=".", help="Path for syne-tune to save finetuning checkpoints."
    )
    parser.add_argument(
        '--lora-target-option',
        default="attn_head_and_mlp",
        help="target module option to apply lora to (names of attn i/p, attn o/p and mlp in LayerAdapter)",
    )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_finetuning_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")


def finetuning_main(args: argparse.Namespace) -> None:
    logging.info("Running SliceGPT post-slicing finetuning experiment")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')


    # This line was commented out during troubleshooting. Uncomment if needed based on memory.
    # if args.sliced_model_path:
    #     # load the sliced model
    #     logging.info(f"Loading sliced {args.model} model from {args.sliced_model_path} with sparsity {args.sparsity}")
    #     model_adapter, tokenizer = hf_utils.load_sliced_model(
    #         args.model,
    #         args.sliced_model_path,
    #         sparsity=args.sparsity,
    #         token=args.hf_token,
    #         round_interval=args.round_interval,
    #     )
    # else:
    #     # load the original model
    #     logging.info(f"Loading {args.model} model")
    #     model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, args.model_path, token=args.hf_token)

    # Based on successful runs, we now know the sliced model is loaded via hf_utils.load_sliced_model
    # If using a sliced model, this branch will be taken:
    if args.sliced_model_path:
        logging.info(f"Loading sliced {args.model} model from {args.sliced_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            token=args.hf_token,
            round_interval=args.round_interval,
        )
    elif args.model_path:
         # Load model from a local path
         logging.info(f"Loading {args.model} model from {args.model_path}")
         # Need to determine the correct adapter for loading from local path if not sliced
         # Assuming hf_utils.get_model_and_tokenizer handles this for standard HF format
         model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, args.model_path, token=args.hf_token)
    else:
        # Load model from Hugging Face Hub
        logging.info(f"Loading {args.model} model from Hugging Face Hub")
        # Assuming hf_utils.get_model_and_tokenizer handles this for standard HF models by name
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, None, token=args.hf_token)


    # get the dataset for perplexity evaluation
    ppl_ds = data_utils.get_dataset(args.ppl_eval_dataset)
    ppl_eval_loader = data_utils.prepare_dataloader(
        dataset=ppl_ds["validation"],
        tokenizer=tokenizer,
        max_seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size,
        nsamples=args.ppl_eval_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    # This block handles model distribution, which helps with OOM
    if args.distribute_model:
        gpu_utils.distribute_model(model_adapter)
    else:
        model_adapter.model.to(config.device) # Ensure model is on specified device if not distributed

    # compute perplexity before finetuning
    # Note: Perplexity evaluation here is on the base (sliced) model before LoRA finetuning
    # The finetuning loss evaluation is handled by the Trainer
    if ppl_eval_loader: # Only evaluate PPL if dataset is loaded
      logging.info(f'PPL before finetuning calculation started...')
      # Ensure model is in evaluation mode for PPL calculation
      model_adapter.model.eval()
      with torch.no_grad(): # No gradients needed for PPL eval
          dataset_ppl = gpu_utils.evaluate_ppl(model_adapter.model, model_adapter.model.config.pad_token_id, ppl_eval_loader)
      logging.info(f'PPL before finetuning: {dataset_ppl:.4f}')
      wandb.log({"pre_finetune_ppl": dataset_ppl})
    else:
      logging.info('PPL evaluation dataset not loaded, skipping pre-finetuning PPL evaluation.')


    utils.cleanup_memory() # Clean up memory after PPL eval

    # get the dataset for finetuning
    finetune_ds = data_utils.get_dataset(args.finetune_dataset)

    # --- Check if finetuning dataset is available and process only if it is ---
    if finetune_ds and "train" in finetune_ds and "test" in finetune_ds:

        finetune_train_loader = data_utils.prepare_dataloader(
            dataset=finetune_ds["train"],
            tokenizer=tokenizer,
            max_seqlen=args.finetune_train_seqlen,
            batch_size=args.finetune_train_batch_size,
            nsamples=args.finetune_train_nsamples,
            varied_seqlen=args.varied_seqlen,
            seed=args.seed,
        )
        finetune_test_loader = data_utils.prepare_dataloader(
            dataset=finetune_ds["test"],
            tokenizer=tokenizer,
            max_seqlen=args.finetune_test_seqlen,
            batch_size=args.finetune_test_batch_size,
            nsamples=args.finetune_test_nsamples,
            varied_seqlen=args.varied_seqlen,
            seed=args.seed,
        )

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            # The lora_target_map function needs to support the specified model
            target_modules=lora_target_map(args.model)[args.lora_target_option],
        )

        model = model_adapter.model # Get the base model again

        # Move model to device before applying PEFT if not distributed
        if not args.distribute_model:
             model.to(config.device) # Ensure base model is on device before wrapping with PEFT

        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()

        # create optimizer and scheduler
        # Use finetune_ds["train"] dataset for length calculation as needed by scheduler
        optimizer, lr_scheduler = get_optimizer_and_scheduler(lora_model, finetune_ds["train"], args)

        # Define Training Arguments
        training_args = TrainingArguments(
            output_dir=args.st_checkpoint_dir,  # output directory
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.finetune_train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.finetune_test_batch_size,  # batch size for evaluation
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            disable_tqdm=False,
            load_best_model_at_end=True, # Load best model based on eval_steps/metric
            eval_steps=args.eval_steps,
            evaluation_strategy=args.evaluation_strategy, # e.g., "steps"
            # metric_for_best_model="eval_loss", # Uncomment and set if you want to save based on a specific metric
            greater_is_better=False,  # Usually True for accuracy, False for loss
            gradient_accumulation_steps=args.gradient_accumulation_steps, # Add accumulation steps
            # gradient_checkpointing=False, # Keep False for debugging OOM related to checkpointing interactions
            # Setting gradient_checkpointing requires model.gradient_checkpointing_enable() which is done below

        )

        # Enable gradient checkpointing *after* wrapping with PEFT and setting training_args
        # Only enable if desired and training_args.gradient_checkpointing is True
        # if training_args.gradient_checkpointing: # Check if it's True in args
        #    lora_model.gradient_checkpointing_enable() # Enable on the model

        # Ensure n_gpu is correctly set for non-distributed training
        if not args.distribute_model:
             training_args._n_gpu = 1
        else:
             # If distributed, accelerate will handle n_gpu
             pass # Let accelerate manage distribution and device placement


        trainer = CustomTrainer(
            model=lora_model,
            tokenizer=tokenizer,
            train_loader=finetune_train_loader, # Use CustomTrainer's dataloaders
            test_loader=finetune_test_loader,
            args=training_args,
            optimizers=(optimizer, lr_scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        )

        # required to enable input grads for gradient_checkpointing if it's True
        # lora_model.enable_input_require_grads() # This is only needed if gradient_checkpointing is True

        # Set model to train mode before training
        lora_model.train()
        logging.info("Starting finetuning training...")
        trainer.train()
        logging.info("Finetuning training finished.")

        # Ensure model is in evaluation mode after training if evaluating PPL again or saving
        lora_model.eval()


        # --- Save the fine-tuned model ---
        if args.save_dir:
            rft_dir = args.save_dir
            if not os.path.exists(rft_dir):
                os.makedirs(rft_dir, exist_ok=True)

            # Save the PEFT adapter weights
            logging.info(f"Saving LoRA adapter weights to {rft_dir}")
            lora_model.save_pretrained(rft_dir)
            logging.info("LoRA adapter weights saved.")

            # You might also want to save the tokenizer and the base model config here
            # The tokenizer is already loaded, just save it
            logging.info(f"Saving tokenizer to {rft_dir}")
            tokenizer.save_pretrained(rft_dir)
            logging.info("Tokenizer saved.")

            # The config of the base model is needed
            logging.info(f"Saving base model config to {rft_dir}")
            model_adapter.model.config.save_pretrained(rft_dir) # Save the config from the sliced base model
            logging.info("Base model config saved.")


            # Optional: If you want to save the *merged* model (base + lora)
            # This is memory intensive and might not be needed if saving adapter is sufficient
            # print("Merging and saving merged model...")
            # merged_model = lora_model.merge_and_unload()
            # merged_model_file = os.path.join(rft_dir, "merged_model.bin") # Or .safetensors
            # torch.save(merged_model.state_dict(), merged_model_file)
            # print(f"Merged model state dict saved to {merged_model_file}")


            # The original script's save logic seems to save the *merged* state dict as a .pt file
            # Let's replicate that if needed, but saving the adapter is standard for PEFT
            # model_file = os.path.join(rft_dir, os.path.basename(args.model) + "_" + str(args.sparsity) + ".pt")
            # logging.info(f"Merging and saving merged model state dict to {model_file}")
            # merged_model_state_dict = lora_model.merge_and_unload().state_dict()
            # torch.save(merged_model_state_dict, model_file)
            # logging.info("Merged state dict saved.")

            # The original script also tries to copy config/tokenizer files from sliced_model_dir
            # Saving tokenizer and config directly above is cleaner
            # if args.sliced_model_path:
            #      sliced_model_dir = args.sliced_model_path
            #      try:
            #          # copy all config files (tokenizer, model and slicing configs)
            #          for file in pathlib.Path(sliced_model_dir).glob("*.json"):
            #              if 'safetensors' not in str(file):
            #                  shutil.copy(str(file), rft_dir)
            #          # copy all tokenizer models
            #          for file in pathlib.Path(sliced_model_dir).glob("*token*.model"):
            #              shutil.copy(str(file), rft_dir)
            #          # copy vocab merges if any
            #          for file in pathlib.Path(sliced_model_dir).glob("merges.txt"):
            #              shutil.copy(str(file), rft_dir)
            #      except OSError as e:
            #          logging.info(f'Failed to copy configs and tokenizer files: {e}')


        logging.info(f"Finetuning process completed. Results saved to {args.save_dir}")

    else:
        logging.info("Finetuning dataset not loaded or available, skipping finetuning.")
        # If not finetuning, you might skip the trainer part and only do PPL eval if args allow


    utils.cleanup_memory() # Clean up memory after finetuning

    # compute perplexity after finetuning (on the lora_model if finetuning happened)
    # Ensure model is in evaluation mode
    model_to_eval_ppl = lora_model if 'lora_model' in locals() and lora_model is not None else model_adapter.model
    model_to_eval_ppl.eval()

    if ppl_eval_loader: # Only evaluate PPL if dataset is loaded
      logging.info(f'PPL after finetuning calculation started...')
      with torch.no_grad(): # No gradients needed for PPL eval
          dataset_ppl = gpu_utils.evaluate_ppl(model_to_eval_ppl, model_adapter.model.config.pad_token_id, ppl_eval_loader)
      logging.info(f'PPL after finetuning: {dataset_ppl:.4f}')
      wandb.log({"post_finetune_ppl": dataset_ppl})

      # Reporter calls are usually for Syne Tune integration
      Reporter()(ppl=dataset_ppl)
      syne_tune.Reporter()(ppl=dataset_ppl)
    else:
        logging.info('PPL evaluation dataset not loaded, skipping post-finetuning PPL evaluation.')


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    finetuning_args = finetuning_arg_parser()
    process_finetuning_args(finetuning_args)

    # Check if CUDA is available and set device if not explicitly specified
    if finetuning_args.device is None:
        finetuning_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convert device string to torch.device object and store in config
    config.device = torch.device(finetuning_args.device)


    finetuning_main(finetuning_args)