from datasets import load_dataset

dataset = load_dataset("emotion", split="train")

emotions = dataset.info.features["label"].names

def query_emotion(start, end):
    rows = dataset[start:end]
    texts, labels = [rows[k] for k in rows.keys()]

    observations = []

    for i, text in enumerate(texts):
        observations.append({
            "text": text,
            "emotion": emotions[labels[i]],
        })

    return observations