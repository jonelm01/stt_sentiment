import tensorflow as tf
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    DataCollatorWithPadding
)
import numpy as np
import random

# Config
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
OUTPUT_DIR = "sentiment-ecommerce-balanced"
BATCH_SIZE = 16
MAX_LENGTH = 300
EPOCHS = 3
SYNTHETIC_POSITIVE_COUNT = 1000  # number of synthetic positives to generate

# Load dataset
dataset = load_dataset("NebulaByte/E-Commerce_Customer_Support_Conversations")

# Label mapping
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

EMOTION_TO_SENTIMENT = {
    "frustrated": "negative",
    "angry": "negative",
    "neutral": "neutral",
    "confused": "neutral",
    "inquiry": "neutral",
    "happy": "positive",
    "satisfied": "positive",
    "pleased": "positive",
    "thankful": "positive",
}

def encode_labels(example):
    raw = example["customer_sentiment"].lower()
    sentiment = EMOTION_TO_SENTIMENT.get(raw, "neutral")
    example["label"] = label2id[sentiment]
    return example

dataset = dataset.map(encode_labels)

# Generate synthetic positive dialogues
positive_templates = [
    "Agent: Hello! How can I help you today?\nCustomer: Hi! I’m really happy with my recent purchase, and I just have a quick question about {product}.\nAgent: Sure! I’d be glad to help you with your {product}.",
    "Agent: Good day! What can I assist you with?\nCustomer: I just wanted to say how satisfied I am with the {product}. Also, I’d like to know how to {action}.\nAgent: Absolutely! Here’s how you can {action}.",
    "Agent: Hi! Thank you for contacting us.\nCustomer: Hello! I’m pleased with my {product} and would like some guidance on {action}.\nAgent: Of course, let me walk you through {action}."
]

products = ["wireless headphones", "StellarSound Pro 3000", "smart speaker", "laptop stand", "gaming mouse"]
actions = ["set up the noise-cancellation", "connect it to my phone", "register for warranty", "update the firmware"]

synthetic_positives = []
for _ in range(SYNTHETIC_POSITIVE_COUNT):
    template = random.choice(positive_templates)
    product = random.choice(products)
    action = random.choice(actions)
    dialogue = template.format(product=product, action=action)
    synthetic_positives.append({"conversation": dialogue, "label": label2id["positive"]})

# Combine with original dataset
from datasets import concatenate_datasets

# Combine original dataset with synthetic positives
synthetic_dataset = Dataset.from_list(synthetic_positives)
all_data = concatenate_datasets([dataset["train"], synthetic_dataset])


# Balance dataset 
def balance_dataset(dataset, label_col="label"):
    counts = {label: sum(1 for x in dataset if x[label_col] == label) for label in [0,1,2]}
    max_count = max(counts.values())

    balanced_data = []
    for label in [0,1,2]:
        samples = [x for x in dataset if x[label_col] == label]
        repeat_factor = max_count // len(samples)
        remainder = max_count % len(samples)
        samples = samples * repeat_factor + samples[:remainder]
        balanced_data.extend(samples)

    np.random.shuffle(balanced_data)
    return Dataset.from_list(balanced_data)

balanced_dataset = balance_dataset(all_data)

# Train / validation split
split = balanced_dataset.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
val_ds = split["test"]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["conversation"], truncation=True, max_length=MAX_LENGTH)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# TensorFlow datasets
data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")

tf_train = train_ds.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols="label",
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

tf_val = val_ds.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols="label",
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

# Model
model = TFAutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Train
model.fit(
    tf_train,
    validation_data=tf_val,
    epochs=EPOCHS,
)

# Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuned balanced 3-class sentiment model with synthetic positives saved at:", OUTPUT_DIR)
