"""
Task 01 - Merge LoRA adapter into base model and export to GGUF
Run AFTER train.py completes.

Steps:
  1. python merge_adapter.py          → saves merged model to ./merged-model/
  2. Manually convert with llama.cpp:
     cd llama.cpp
     python convert_hf_to_gguf.py ../merged-model --outtype q4_k_m \
         --outfile ../phi3-devops-q4.gguf
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_DIR = "./lora-adapter"
MERGED_DIR = "./merged-model"


def merge_and_save():
    Path(MERGED_DIR).mkdir(exist_ok=True)

    print("Loading base model in 4-bit for merging...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}/")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print("\nDone! Next step — convert to GGUF:")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  cd llama.cpp && pip install -r requirements.txt")
    print("  python convert_hf_to_gguf.py ../merged-model \\")
    print("      --outtype q4_k_m --outfile ../phi3-devops-q4.gguf")


if __name__ == "__main__":
    merge_and_save()
