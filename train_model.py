from datasets import load_dataset
import torch
#from unsloth import FastLanguageModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import bitsandbytes as bnb
import matplotlib.pyplot as plt
from transformers import TrainerCallback


print("loading model")
# model_name = "Qwen/Qwen2.5-0.5B-Instruct" 
model_name = "microsoft/phi-2"
#better to do SFT on bias prompt output examples
#before doing DPO
#(so we don't just use the base model)

model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    r=16, #rank of the matrices. smaller makes training faster
    lora_alpha=32, #scaling factor. higher increases the influence of LoRA matrices
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.1,
    # target_modules=["q_proj", "v_proj", ...]  # optionally indicate target modules
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# now perform training on your dataset, e.g. using transformers Trainer, then save the model
#model.save_pretrained("qwen2.5-3b-lora")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token #set pad token
tokenizer.padding_side = "left"


ds = load_dataset("ahmedallam/BiasDPO")
#print(ds)
#print(ds["train"]) #this is a HF dataset

ds = ds["train"]


# def chatml_format(example):
#     """
#     Formats a dataset sample into a structure suitable for training
#     by wrapping it with ChatML template.

#     Parameters:
#     example (dict): A dictionary containing the following keys:
#         - 'system' (str): System message content.
#         - 'question' (str): User's question or instruction.
#         - 'chosen' (str): The chosen response.
#         - 'rejected' (str): The rejected response.

#     Returns:
#     dict: A dictionary with the following keys:
#         - 'prompt' (str): Combined system and user prompt formatted for training.
#         - 'chosen' (str): The chosen response with an end-of-message marker.
#         - 'rejected' (str): The rejected response with an end-of-message marker.
#     """
#     # Format system
#     # if len(example['system']) > 0:
#     #     message = {"role": "system", "content": example['system']}
#     #     system = tokenizer.apply_chat_template([message], tokenize=False)
#     # else:
#     system = ""

#     # Format instruction
#     message = {"role": "user", "content": example['prompt']}
#     prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

#     # Format chosen answer
#     chosen = example['chosen'] + "<|im_end|>\n"

#     # Format rejected answer
#     rejected = example['rejected'] + "<|im_end|>\n"

#     return {
#         "prompt": system + prompt,
#         "chosen": chosen,
#         "rejected": rejected,
#     }

# # Wrap dataset in ChatML prompt
# original_columns = ds.column_names
# ds = ds.map(
#     chatml_format,
#     remove_columns=original_columns
# )

ds_split = ds.train_test_split(test_size=0.1, seed=42)
train_dataset = ds_split["train"]
val_dataset = ds_split["test"]

print("Number of training data samples:")
print(len(train_dataset))

# Use a pipeline as a high-level helper

# Print sample
print("Sample data")
train_dataset[1]


# Enable Apple GPU (MPS) if supported
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

if device == "cpu":
    print("Using CPU, MPS not supported")


# Define training arguments
# training_args = DPOTrainingArguments(
#     per_device_train_batch_size = 2,   # Number of samples per GPU/CPU core
#     gradient_accumulation_steps = 4,   # Accumulate gradients over 4 steps before optimizer update => simulates batch size of 8
#     max_steps = 50,                    # Stop training after 50 update steps
#     learning_rate = 5e-5,              # Initial learning rate for optimizer
#     logging_steps = 1,                 # Log metrics every 1 step
#     fp16 = False,                      # Mixed precision with FP16 — disabled here (MPS doesn't support it)
#     bf16 = False,                      # BFloat16 precision — disabled too (MPS doesn't support it)
#     optim = "adamw_torch",             # Use standard AdamW optimizer from PyTorch (not 8-bit or fused)
#     output_dir = "outputs",            # Where checkpoints and logs will be saved
#     report_to = "none",                # Disable logging to WandB or other services
# )
training_args = DPOConfig(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2, #simulates a larger batch size
    #effective batch size is 2 * 2 = 4
    #max_steps=100,
    #num_train_epochs=5,
    learning_rate=1e-6,
    logging_steps=10,  # Log progress every step
    save_steps=50,
    use_mps_device = True,
    #fp16=False,
    bf16=False,
    report_to="none",
    max_steps=200,
    beta=0.1,)

# Define DPO trainer
trainer = DPOTrainer(
    model = model,
    ref_model = AutoModelForCausalLM.from_pretrained(model_name),  # Pass a copy of the base model as the reference model
    args = training_args,
    train_dataset = train_dataset,   # Your dataset with `prompt`, `chosen`, `rejected` fields
    eval_dataset = val_dataset,
    processing_class = tokenizer,           # Pass the tokenizer directly
   
)

# # Print progress callback
# def print_progress_callback(step, logs):
#     print(f"Step {step}: {logs}")

# trainer.add_callback(print_progress_callback)

print("training")

# Define a callback to log reward margins
class RewardMarginLoggerCallback(TrainerCallback):
    def __init__(self):
        self.reward_margins = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward_margin" in logs:
            self.reward_margins.append(logs["reward_margin"])

# Instantiate the callback
reward_margin_logger = RewardMarginLoggerCallback()

# Add the callback to the trainer
trainer.add_callback(reward_margin_logger)

# Start training
trainer.train()

# After training, plot the reward margins
print("Training complete. Plotting reward margins...")

# Plot the reward margins
plt.figure(figsize=(10, 6))
plt.plot(reward_margin_logger.reward_margins, label="Reward Margin")
plt.xlabel("Training Steps")
plt.ylabel("Reward Margin")
plt.title("Reward Margin During Training")
plt.legend()
plt.grid()
plt.savefig("outputs/reward_margins.png")  # Save the plot
plt.show()

