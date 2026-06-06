from pathlib import Path
from src.chunking import ChunkingStrategyComparator

comparator = ChunkingStrategyComparator()

documents = [
    "Paper/paper1.md",
    "Paper/paper2.md",
    "Paper/paper3.md"
]

for doc in documents:
    text = Path(doc).read_text(encoding="utf-8")

    result = comparator.compare(text)

    print(f"\n===== {doc} =====")
    print("Fixed:", result["fixed_size"]["count"])
    print("Sentence:", result["by_sentences"]["count"])
    print("Recursive:", result["recursive"]["count"])
    
    for strategy, stats in result.items():
        print(
            f"{strategy}: "
            f"count={stats['count']}, "
            f"avg_length={stats['avg_length']:.2f}"
        )

