import math
import os

import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup


class TrainingLoops:
    def __init__(
        self,
        model,
        dataset=None,
        training_portion=0.8,
        validation_portion=0.2,
        batch_size=1,
        epochs=5,
        lr=1e-4,
        gradient_accumulation_steps=16,
        output_dir="./checkpoints",
        trainset=None,
        valset=None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Exactly one source: either a whole dataset to split, or a train/val pair.
        if dataset is not None and (trainset is not None or valset is not None):
            raise ValueError("Provide a whole dataset OR a trainset/valset pair, not both")
        if trainset is not None or valset is not None:
            if trainset is None or valset is None:
                raise ValueError("Both trainset and valset are required")
            if len(trainset) == 0 or len(valset) == 0:
                raise ValueError("Train and validation datasets must both be nonempty")
            self.train_set = trainset
            self.validation_set = valset
            self.total_dataset_size = len(trainset) + len(valset)
        elif dataset is not None:
            # One dataset, with a deterministic train/validation split.
            if len(dataset) < 2:
                raise ValueError("Need at least 2 examples to create train and validation sets")
            if not (0 < training_portion < 1 and 0 < validation_portion < 1):
                raise ValueError("Train/validation fractions must each be between 0 and 1")
            if not math.isclose(training_portion + validation_portion, 1.0, abs_tol=1e-6):
                raise ValueError("Train and validation fractions must sum to 1")
            # Integer split ensures both splits are nonempty, even for small datasets.
            train_count = max(1, min(len(dataset) - 1, int(len(dataset) * training_portion)))
            val_count = len(dataset) - train_count
            self.train_set, self.validation_set = random_split(
                dataset, [train_count, val_count], generator=torch.Generator().manual_seed(42)
            )
            self.total_dataset_size = len(dataset)
        else:
            raise ValueError("Provide either dataset or trainset")

        if len(self.train_set) == 0:
            raise ValueError("Training dataset is empty")
        self.dataset = self.train_set  # For sample_train() compatibility.
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        # Validation is mandatory, so always create its DataLoader.
        self.validation_loader = DataLoader(
            self.validation_set, batch_size=self.batch_size, shuffle=False
        )

        trainable_parameters = [p for p in self.model.model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters; check the LoRA/model configuration")
        self.optimizer = torch.optim.AdamW(trainable_parameters, lr=self.lr)

        updates_per_epoch = math.ceil(len(self.train_loader) / self.gradient_accumulation_steps)
        total_updates = updates_per_epoch * self.epochs
        warmup_steps = int(0.03 * total_updates)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
        self.global_step = 0
        self._last_saved_step = -1

        print("\n===== Training Configuration =====")
        # print(f"Trainable parameters:     {self.model.model.print_trainable_parameters()}")
        print(f"Model: {self.model.model_name}")
        print(f"Total dataset size:       {self.total_dataset_size}")
        print(f"Training samples:         {len(self.train_set)}")
        print(f"Validation samples:       {len(self.validation_set)}")
        print(f"Batch size:               {self.batch_size}")
        print(f"Gradient accumulation:    {self.gradient_accumulation_steps}")
        print(f"Effective batch size:     {self.batch_size * self.gradient_accumulation_steps}")
        print(f"Epochs:                   {self.epochs}")
        print(f"Learning rate:            {self.lr}")
        print(f"Train batches per epoch:  {len(self.train_loader)}")
        print(f"Optimizer updates/epoch:  {updates_per_epoch}")
        print(f"Total optimizer updates:  {total_updates}")
        print(f"Warmup steps:             {warmup_steps}")
        print("==================================\n")

    def sample_train(self):
        sample = self.train_set[0]
        for epoch in range(2):
            total_loss = 0.0
            for i in range(20):
                loss = self.model.train_step(sample["question"], sample["answer"])
                total_loss += float(loss)
                print(f"Sample: Step {i + 1}: {float(loss):.4f}")
            print(f"Sample: Average loss: {total_loss / 20:.4f}")

    def validate(self):
        print("\nRunning validation...")
        self.model.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in self.validation_loader:
                questions = batch["question"]
                answers = batch["answer"]
                systems = batch.get("system", [""] * len(questions))
                loss = self.model.calculate_batch_loss(questions=questions,answers=answers,systems=systems)
                count = len(questions)
                total_loss += loss.item() * count
                total_samples += count
        return total_loss / total_samples

    def train(self):
        for epoch in range(self.epochs):
            print(f"\n{'=' * 50}\nStarting Epoch {epoch + 1}/{self.epochs}\n{'=' * 50}")
            self.model.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_examples = 0

            for batch_idx, batch in enumerate(self.train_loader):
                print(f"[Epoch {epoch + 1}/{self.epochs}] Batch {batch_idx + 1}/{len(self.train_loader)}")
                questions = batch["question"]
                answers = batch["answer"]
                systems = batch.get("system", [""] * len(questions))

                loss = self.model.calculate_batch_loss(questions= questions, answers= answers, systems=systems)
                print(f"  Raw batch loss: {loss.item():.4f}")
                current_batch_size = len(questions)
                total_loss += loss.item() * current_batch_size
                total_examples += current_batch_size

                # Scale by actual example count in this accumulation window;
                # this also handles the partial window at the end of an epoch.
                group_start_batch = (batch_idx // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
                remaining_examples = len(self.train_set) - group_start_batch * self.batch_size
                window_examples = min(self.gradient_accumulation_steps * self.batch_size, remaining_examples)
                scaled_loss = loss * (current_batch_size / window_examples)
                scaled_loss.backward()

                should_update = (batch_idx + 1) % self.gradient_accumulation_steps == 0
                is_last_batch = (batch_idx + 1) == len(self.train_loader)
                if should_update or is_last_batch:
                    print("  -> Optimizer update")
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f"  -> Global step: {self.global_step} | LR: {current_lr:.8f}")
                    if self.global_step % 10 == 0:
                        print(f"Epoch {epoch + 1}/{self.epochs} | Step {self.global_step} | Loss {loss.item():.4f}")
                    if self.global_step % 50 == 0:
                        self.save_checkpoint(epoch)
                else:
                    print("  -> Accumulating gradients")

            average_train_loss = total_loss / total_examples
            validation_loss = self.validate()
            print("\n" + "=" * 50)
            print(f"Epoch {epoch + 1}/{self.epochs} Complete")
            print(f"Average training loss:   {average_train_loss:.4f}")
            print(f"Average validation loss: {validation_loss:.4f}")
            print(f"Global optimizer steps:  {self.global_step}")
            print("=" * 50)
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        mode_name = self.model.model_name
        if self._last_saved_step == self.global_step:
            return
        checkpoint_dir = os.path.join(self.output_dir, f"{mode_name}_checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.model.save_pretrained(checkpoint_dir)
        self.model.tokenizer.save_pretrained(checkpoint_dir)
        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        self._last_saved_step = self.global_step
        print(f"Checkpoint saved at: {checkpoint_dir}")
