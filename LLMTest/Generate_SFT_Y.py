import json

input_path = "output.txt"
output_path = "sft_dataset.jsonl"

examples = []
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

for p in paragraphs:
    prompt = p
    # Replace 'thou' -> 'the' case-insensitive
    completion = prompt.replace("thou", "you").replace("Thou", "You")
    if prompt != completion:
        examples.append({
            "prompt": prompt,
            "completion": completion
        })

print(f"Created {len(examples)} SFT examples.")

# Save as JSONL
with open(output_path, "w", encoding="utf-8") as fout:
    for ex in examples:
        fout.write(json.dumps(ex) + "\n")

print(f"SFT dataset saved to {output_path}")