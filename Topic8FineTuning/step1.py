import json, random

with open("sql_create_context_v4.json") as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print(f"\nSample example:")
ex = data[100]
print(f"  Question: {ex['question']}")
print(f"  Context:  {ex['context'][:120]}...")
print(f"  Answer:   {ex['answer']}")


NUM_TEST_EXAMPLES = 200  # Held-out for evaluation; all remaining data used for training
random.shuffle(data)
test_data = data[:NUM_TEST_EXAMPLES]
train_data = data[NUM_TEST_EXAMPLES:]
print(f"Training examples: {len(train_data)} (all except evaluation)")
print(f"Test examples: {len(test_data)}")


with open("train_data.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("test_data.json", "w") as f:
    json.dump(test_data, f, indent=2)

print("Saved train_data.json and test_data.json")