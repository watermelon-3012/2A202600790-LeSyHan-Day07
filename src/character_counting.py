from pathlib import Path

content = Path("Paper/paper5.md").read_text(encoding="utf-8")

num_chars = len(content.replace(" ", "").replace("\n", "").replace("\t", ""))

print(f"Characters (excluding whitespace): {num_chars}")
