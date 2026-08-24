from datasets import load_dataset

print("Loading MedDialog English dataset...")

dataset = load_dataset(
    "UCSD26/medical_dialog",
    name="en"
)

print("\nDataset loaded successfully!")

print(dataset)

print("\nTrain examples:")
print(dataset["train"].num_rows)

print("\nFirst conversation:")
print(dataset["train"][0])