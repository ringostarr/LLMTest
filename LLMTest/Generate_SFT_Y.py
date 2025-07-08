examples = []
with open("output.txt", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.split("\n\n")  # or split however makes sense

for line in lines:
    prompt = line.strip()
    # Replace "thou" -> "the" (case insensitive):
    completion = prompt.replace("thou", "the").replace("Thou", "The")
    if prompt != completion:  # only keep if there's a change
        examples.append({
            "prompt": prompt,
            "completion": completion
        })
from datasets import Dataset

dataset = Dataset.from_list(examples)
print(dataset[0])