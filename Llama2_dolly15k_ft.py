# %%
!nvidia-smi

# %%
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, PeftModel, LoraConfig, get_peft_model


# %%
from datasets import load_dataset

train_data , test_data =load_dataset("databricks/databricks-dolly-15k", split=["train[:80%]", "train[80%:]"])

# %%
import pandas as pd

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# %%
train_df = train_df[train_df["context"].str.len()>=10]
test_df = test_df[test_df["context"].str.len()>=10]
train_df.reset_index(drop=True,inplace=True)
test_df.reset_index(drop=True,inplace=True)


# %%
train_df

# %%
train_df.shape, test_df.shape

# %%
def prepare_dataset(df, split="train"):
    text_col=[]
    instruction="""write a precise summary of the below input text. 
    Return your response in bullet points which covers the keypoints of the input text.
    only provide full sentences response summary."""
    if split == "train":
        for _, row in df.iterrows():
            inst=row["instruction"]
            inputc=row["context"]
            output=row["response"]
            text = ("### Instruction: \n" + instruction +"\n" + inst
                    +"\n### Input: \n" + inputc + "\n### Response: \n" + output)
            text_col.append(text)
        df.loc[:, "text"] = text_col
    else:
        for _, row in df.iterrows():
            inst=row["instruction"]
            inputc=row["context"]
            text = ("### Instruction: \n" + instruction +"\n" + inst
                    +"\n### Input: \n" + inputc + "\n### Response: \n" )
            text_col.append(text)
        df.loc[:, "text"] = text_col
    return df    

# %%
train_df = prepare_dataset(train_df, "train")
test_df = prepare_dataset(test_df, "test")

# %%
print(train_df["text"][0])

# %%
print(test_df["text"][0])

# %%
from datasets import Dataset
dataset = Dataset.from_pandas(train_df)
dataset

# %%
model_name = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type="float16"
)
model =AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,trust_remote_code=True,device_map="auto")
model.config.use_cache = True

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,return_token_type_ids=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# %%
#Lora configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# %%
# training args & prepare model for kbit training
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training
from time import perf_counter


args = TrainingArguments(
    output_dir="./llama2_ft_dolly",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=1,
    max_steps=100,
    fp16=True,
    push_to_hub=False,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# %%
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    args=args,
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=512
)

# %%
start_time = perf_counter()
trainer.train()
end_time = perf_counter()
training_time = end_time - start_time
print(f"Time taken for training: {training_time}")

# %%
model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
model_to_save.save_pretrained("./llama2_ft_dolly/results")

# %%
lora_config = LoraConfig.from_pretrained("./llama2_ft_dolly/results")
tmodel = get_peft_model(model_to_save, lora_config)

# %%
from transformers import GenerationConfig
start_time = perf_counter()
text = test_df["text"][5]

inputs = tokenizer(text, return_tensors="pt").to("cuda")
generation_config = GenerationConfig(
    penalty_alpha=0.6, do_sample=True, top_k=5, temperature=0.5, repetition_penalty=1.2)
outputs = tmodel.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    generation_config=generation_config,
    max_new_tokens=100,)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
end_time = perf_counter()
output_time = end_time - start_time
print(f"Time taken for output: {output_time} seconds")

# %%
text1 = test_df["text"][10]
inputs = tokenizer(text1, return_tensors="pt").to("cuda")
generation_config = GenerationConfig(
    penalty_alpha=0.6, do_sample=True, top_k=5, temperature=0.5, repetition_penalty=1.2)
outputs = tmodel.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    generation_config=generation_config,
    max_new_tokens=100,)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
end_time = perf_counter()
output_time = end_time - start_time
print(f"Time taken for output: {output_time} seconds")

# %%
!nvidia-smi

# %%
from transformers import GPTQConfig

quantization_config = GPTQConfig(bits=4,dataset=["c4"],desc_act=False)
quant_model = AutoModelForCausalLM.from_pretrained("./llama2_ft_dolly/outputs", quantization_config=quantization_config,device_map="auto")

# %%
# saving the quantized model
quant_model.save_pretrained("./llama2_ft_dolly/quantized", safe_serialization=True)
tokenizer.save_pretrained("./llama2_ft_dolly/quantized")

# %%
start_time = perf_counter()
test = test_df["text"][5]
inputs = tokenizer(test, return_tensors="pt").to("cuda")
gen_config = GenerationConfig(
    penalty_alpha=0.6, do_sample=True, top_k=5, temperature=0.5, repetition_penalty=1.2)
outputs = quant_model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    generation_config=gen_config,
    max_new_tokens=100,)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
end_time = perf_counter()
output_time = end_time - start_time
print(f"Time taken for output: {output_time} seconds")

# %%
start_time = perf_counter()
test1 = test_df["text"][10]
inputs = tokenizer(test1, return_tensors="pt").to("cuda")
gen_config = GenerationConfig(
    penalty_alpha=0.6, do_sample=True, top_k=5, temperature=0.5, repetition_penalty=1.2)
outputs = quant_model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    generation_config=gen_config,
    max_new_tokens=100,)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
end_time = perf_counter()
output_time = end_time - start_time
print(f"Time taken for output: {output_time} seconds")

# %%



