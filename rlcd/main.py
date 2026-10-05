import random
from helper import hf_login, load_dataset
from processing import prepare_xlam_for_laya, prepare_dataset_items
from training import RLTrainingConfig, RLTrainer

#### Model variables ####
MODEL_NAME_OR_PATH="convaiinnovations/laya"
OUTPUT_DIR="./laya_finetuned_xlam_rl"
#########################

#### Dataset variables ####
DATASET = "Salesforce/xlam-function-calling-60k"
DATASET_SUBSET = 100 # set to None to use full dataset
###########################

#### Environment variables ####
DEVICE="cpu" # Change to "cuda" if testing locally with a GPU (Requires CUDA-enabled torch)
###############################

#### Training variables ####
EPOCHS=3
MICRO_BATCH_SIZE=8
GRAD_ACCUM_STEPS=4
REWARD_STRATEGY="brier"
CALIB_MAX=10 # Using 10 items for calibration for demonstration
############################

def main():
    """
    End-to-end script to fine-tune Laya on the Salesforce XLAM dataset using RL.
    """
    # 1. Authenticate with Hugging Face (uses HF_TOKEN from .env if present)
    hf_login()

    # 2. Load the dataset
    print(f"\n===Loading '{DATASET}' dataset===")
    dataset = load_dataset(DATASET)

    if DATASET_SUBSET is not None:
        train_data = dataset["train"].select(range(DATASET_SUBSET))
    else:
        train_data = dataset["train"]

    # 3. Preprocess the dataset
    print("\n===Preprocessing dataset into Laya format===")
    prepared_dataset = prepare_xlam_for_laya(train_data)
    
    # 4. Configure the RL Training parameters
    print("\n===Configuring the RL Trainer===")
    config = RLTrainingConfig(
        model_name_or_path=MODEL_NAME_OR_PATH,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        micro_batch_size=MICRO_BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        reward_strategy=REWARD_STRATEGY,
        device=DEVICE,
        calib_max=CALIB_MAX
    )
    
    # Initialize the Trainer to get the tokenizer
    # This will automatically download/load the base model using models.py
    trainer = RLTrainer(config=config)
    
    print("\n===Tokenizing sequences===")
    all_items = prepare_dataset_items(
        prepared_dataset,
        trainer.tokenizer,
        max_len=config.max_len,
        head_max_len=config.head_max_len,
    )

    # Withhold calibration slice with fixed seed
    order = list(range(len(all_items)))
    random.Random(config.calib_seed).shuffle(order)
    n_calib = min(config.calib_max, max(10, int(len(all_items) * config.calib_ratio)))

    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]

    # 6. Run the RL loop!
    print("\n===Starting Reinforcement Learning fine-tuning===")
    final_model_path = trainer.train(train_items=train_items, calib_items=calib_items)

if __name__ == "__main__":
    main()
