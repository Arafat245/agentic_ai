import tinker
import json

service_client = tinker.ServiceClient()
base_model = "meta-llama/Llama-3.2-1B"
training_client = service_client.create_lora_training_client(base_model=base_model)
tokenizer = training_client.get_tokenizer()

print("\n--- Evaluating Base Model on 200 Test Questions ---")
base_sampling_client = training_client.save_weights_and_get_sampling_client(
    name="base-model"
)

with open("test_data.json", "r") as f:
    test_data = json.load(f)

base_accuracy = evaluate_test_set(
    base_sampling_client, tokenizer, test_data, "base"
)
print(f"Base model accuracy: {base_accuracy:.2%} ({int(base_accuracy * 200)}/200)")