import os
from dotenv import load_dotenv
import datasets
from huggingface_hub import login

load_dotenv()

def hf_login():
    """Authenticate with the Hugging Face Hub using a token from the environment.

    """
    # Authenticate with Hugging Face if HF_TOKEN is defined
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token != "your_huggingface_token_here":
        login(token=hf_token)
        print("Logged in with HF_TOKEN from .env file.")
    else:
        print("No HF_TOKEN found in .env; using cached local Hugging Face credentials if available.")

def load_dataset(dataset_name=None):
    """Load a dataset from the Hugging Face Hub and print a summary of it.

    Args:
        dataset_name (str, optional): Name/identifier of the dataset on the Hugging
            Face Hub (e.g. ``"org/dataset"``), passed directly to
            ``datasets.load_dataset``.

    Returns:
        datasets.DatasetDict or datasets.Dataset: The loaded dataset object,
        as returned by ``datasets.load_dataset``.
    """

    if dataset_name is None:
        dataset_name = os.getenv("DEFAULT_DATASET_NAME")

    dataset = datasets.load_dataset(dataset_name)

    print(f"Loaded {dataset_name}...")

    return dataset

