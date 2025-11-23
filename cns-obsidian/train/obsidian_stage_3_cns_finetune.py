import time
import wandb
from tqdm import tqdm
from PIL import Image
from functools import partial

import torch
import torch.distributed as dist  
# TODO move is_main_process(): to utils
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision

from transformers import LlavaNextProcessor
from transformers import LlavaNextForConditionalGeneration
from transformers.models.clip.modeling_clip import CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextMultiModalProjector,
)
from transformers import get_cosine_schedule_with_warmup

from cns_obsidian.utils import setup_distributed
from cns_obsidian.utils import kill_distributed
from cns_obsidian.utils import save_model_distributed
from cns_obsidian.utils import count_parameters
from cns_obsidian.datasets import CNSDataModule

# Set float32 matmul precision for Tensor Cores
torch.set_float32_matmul_precision("high")

Image.MAX_IMAGE_PIXELS = 10000000000

# Hyperparameters
# # TODO obviously these all should be config / args
hyperparameters = {
    "max_length": 4096,
    # unfreezing (optimizer exclusion) choices
    "unfreeze_language_model": True,
    "unfreeze_vision_tower": False,
    # optimizer's parameters
    "learning_rate": 1e-5,
    "learning_rate_vision": None,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.00,
    # training length and batch sizes
    "epochs": 5,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "validation_interval": 1000,
    # warm-up
    "warmup_ratio": 0.03,
    # data loaders num workers
    "num_workers": 16,
    # saving the model
    "checkpoints_dir": "/gpfs/data/oermannlab/users/alyaka01/checkpoints/obsidian/stage_3_both_fix_05_10_10",
    # loading the model
    "base_model_path": "/gpfs/data/oermannlab/users/alyaka01/.cache/huggingface/hub/models--llava-hf--llava-v1.6-34b-hf/snapshots/982f7ce5c212963a57284eefd509cc7cb4e4376e",
    "checkpoint_path": "/gpfs/data/oermannlab/users/alyaka01/checkpoints/obsidian/stage_2_05_10/checkpoint_9.pt",
}


def initialize_model(
    unfreeze_language=False, unfreeze_vision=False, hyperparameters={}
):
    # Load the model on CPU
    model = LlavaNextForConditionalGeneration.from_pretrained(
        hyperparameters["base_model_path"],
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # Load the parameters into the model
    if hyperparameters["checkpoint_path"]:
        checkpoint = torch.load(hyperparameters["checkpoint_path"])

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    # Set exclude_from_optimizer only for specific parts based on the flag
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True
        elif "language_model" in name:
            param.requires_grad = unfreeze_language
        elif "vision_tower" in name:
            param.requires_grad = unfreeze_vision
        else:
            print(f"We have other params: {name}")
            param.requires_grad = False

    # might be confusing, but this is function that can wrap both ac and fsdp
    # TODO in the newer torch this can be substituted with transformer policy
    def custom_wrap_policy(
        module, recurse=False, nonwrapped_numel=-1, include_model_pieces=False
    ):
        transformer_layer_cls = {
            CLIPEncoderLayer,
            LlamaDecoderLayer,
        }
        model_pieces_cls = {
            # CLIPVisionModel # TODO this broke but i haven't tried with 2.5
            LlavaNextMultiModalProjector,
            # LlamaForCausalLM
        }

        cls_to_wrap = transformer_layer_cls | (
            model_pieces_cls if include_model_pieces else set()
        )

        if recurse:
            return True
        else:
            return isinstance(module, tuple(cls_to_wrap))

    fsdp_wrap_policy = partial(custom_wrap_policy, include_model_pieces=True)

    # Define mixed precision policy
    # TODO technically this shouldn't change anything if we load in bfloat16
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # wrap with fsdp
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_wrap_policy,
        device_id=torch.cuda.current_device(),
        mixed_precision=mixed_precision_policy,
        use_orig_params=True,
    )

    # imports need to happen here even though i dont understand why
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    # checkpoint wrap (very similar to what fabric does internally)
    checkpoint_wrapper = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=custom_wrap_policy,
    )

    # print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - State of the Model b4 train")
    # print(model)
    return model


def create_optimizer(model, hyperparameters):
    # Define learning rates for each group
    lr_language = hyperparameters["learning_rate"]
    lr_projector = hyperparameters["learning_rate"]
    lr_vision = hyperparameters.get(
        "learning_rate_vision", hyperparameters["learning_rate"]
    )

    # Create parameter groups
    vision_params = []
    language_params = []
    projector_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "multi_modal_projector" in name:
                projector_params.append((name, param))
            elif "language_model" in name:
                language_params.append((name, param))
            elif "vision_tower" in name:
                vision_params.append((name, param))
            else:
                # image newline will go with other image params
                other_params.append((name, param))

    # we are always optimizing the projection
    param_groups = [
        {
            "name": "projector",
            "params": [p for _, p in projector_params],
            "lr": lr_projector,
        }
    ]

    # include the language model in param groups if we are optimizing it
    if hyperparameters["unfreeze_language_model"]:
        param_groups.append(
            {
                "name": "language",
                "params": [p for _, p in language_params],
                "lr": lr_language,
            }
        )

    # include the vision model in param groups if we are optimizing it
    if hyperparameters["unfreeze_vision_tower"]:
        param_groups.append(
            {
                "name": "vision",
                "params": [p for _, p in vision_params],
                "lr": lr_vision,
            }
        )

    # Create optimizer with parameter groups
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=hyperparameters["betas"],
        eps=hyperparameters["eps"],
        weight_decay=hyperparameters["weight_decay"],
    )

    return optimizer


def train(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    epochs=1,
):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Starting Training")
    torch.cuda.empty_cache()
    model.train()

    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Starting Training Epoch {epoch}"
        )
        model.train()
        total_loss = 0.0
        for iteration, batch in enumerate(
            tqdm(train_dataloader, desc="Training", leave=False)
        ):
            outputs = model(**batch)
            loss = outputs.loss
            # Normalize the loss to account for gradient accumulation
            loss = loss / hyperparameters["gradient_accumulation_steps"]
            total_loss += loss.item()

            # Scale the loss and call backward
            loss.backward()

            # Log the loss and learning rates
            lr_log = {"train/loss_step": loss.item(), "step": iteration}
            for i, group in enumerate(optimizer.param_groups):
                lr_log[f"learning_rate_group_{i}"] = group["lr"]
            if dist.get_rank() == 0:
                wandb.log(lr_log)

            # Perform optimization step after accumulating gradients
            if (iteration + 1) % hyperparameters[
                "gradient_accumulation_steps"
            ] == 0:
                # Check for NaN or Inf gradients
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if (
                            torch.isnan(param.grad).any()
                            or torch.isinf(param.grad).any()
                        ):
                            print(
                                f"NaN or Inf detected in gradients for {name}"
                            )
                            valid_gradients = False
                            break

                if valid_gradients:
                    optimizer.step()
                else:
                    print(
                        f"Skipping optimizer step at iteration {iteration} due to invalid gradients"
                    )
                scheduler.step()
                optimizer.zero_grad()

                # save every n iterations
                if (iteration + 1) % hyperparameters[
                    "validation_interval"
                ] == 0:
                    # Save the model every few steps
                    epoch_iteration = epoch + iteration / len(train_dataloader) - 1
                    save_model_distributed(
                        model,
                        optimizer,
                        scheduler,
                        epoch_iteration,
                        loss.item(),
                        checkpoints_dir=hyperparameters["checkpoints_dir"],
                    )

        # Perform any remaining optimization step
        if (
            len(train_dataloader)
            % hyperparameters["gradient_accumulation_steps"]
            != 0
        ):
            # Check for NaN or Inf gradients
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if (
                        torch.isnan(param.grad).any()
                        or torch.isinf(param.grad).any()
                    ):
                        print(f"NaN or Inf detected in gradients for {name}")
                        valid_gradients = False
                        break
            if valid_gradients:
                optimizer.step()
            else:
                print(
                    f"Skipping final optimizer step of epoch {epoch} due to invalid gradients"
                )

            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Epoch {epoch} completed. Average train loss: {avg_train_loss}"
        )

        save_model_distributed(
            model,
            optimizer,
            scheduler,
            epoch,
            avg_train_loss,
            checkpoints_dir=hyperparameters["checkpoints_dir"],
        )
        if dist.get_rank() == 0:
            wandb.log({"train/loss_epoch": avg_train_loss, "epoch": epoch})

        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Starting Validation for Epoch {epoch}"
        )
        validate(model, val_dataloader, epoch, iteration)


def validate(model, val_dataloader, epoch, iteration):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating", leave=False):
            outputs = model(**batch)
            loss = outputs.loss
            total_val_loss += loss.item()
            lr_log = {"val/loss_step": loss.item(), "step": iteration}
            if dist.get_rank() == 0:
                wandb.log(lr_log)

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Validation completed. Average validation loss: {avg_val_loss}"
    )
    if dist.get_rank() == 0:
        wandb.log(
            {"val/loss_epoch": avg_val_loss, "epoch": epoch, "step": iteration}
        )
    model.train()


def main():
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Let's fucking gooooo!")
    setup_distributed()

    # Initialize wandb
    if dist.get_rank() == 0:
        wandb.init(
            entity="alyakin314",
            project="obsidian-stage-3-cns-finetune-only-gpt",
            # project="obsidian-stage-3-cns-finetune-only-claude",
            # project="obsidian-stage-3-cns-finetune-both",
            config=hyperparameters,
            dir="~/wandb",
        )

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Initializing Processor")
    processor = LlavaNextProcessor.from_pretrained(
        hyperparameters["base_model_path"],
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "right"

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Initializing Model")
    model = initialize_model(
        hyperparameters=hyperparameters,
        unfreeze_language=hyperparameters["unfreeze_language_model"],
        unfreeze_vision=hyperparameters["unfreeze_vision_tower"],
    )

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Setting up optimizer")
    optimizer = create_optimizer(model, hyperparameters)

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Creating data module")
    datamodule = CNSDataModule(
        "/gpfs/data/oermannlab/private_data/TheMedScrolls/FiguresJadenTextract",
        processor,
        # dataset_file="full_journal_dataset.json",
        # dataset_file="full_journal_dataset_claude.json",
        # dataset_file="full_journal_dataset_both.json",
        dataset_file="full_journal_dataset_both_fix.json",
        data_key="conversations",
        batch_size=hyperparameters["per_device_batch_size"],
        max_tokens=hyperparameters["max_length"],
        paper_level_split="exists",
        distributed=True,
        ift=True,
    )
    # f
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # Calculate total number of training steps
    total_steps = (
        len(train_dataloader)
        * hyperparameters["epochs"]
        // hyperparameters["gradient_accumulation_steps"]
    )
    warmup_steps = int(total_steps * hyperparameters["warmup_ratio"])

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Initializing scheuler")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    save_model_distributed(
        model,
        optimizer,
        scheduler,
        -1,
        -1,
        checkpoints_dir=hyperparameters["checkpoints_dir"],
    )

    if dist.get_rank() == 0:
        wandb.watch(model, log="all", log_freq=1)

    train(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        hyperparameters["epochs"],
    )

    kill_distributed()


if __name__ == "__main__":
    main()
