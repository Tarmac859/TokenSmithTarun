"""
Converts extracted_sections.json back into the markdown format
that the TokenSmith indexer expects, so we can skip re-running
the heavy docling extraction.
"""
import json
from pathlib import Path

input_path = Path("data/extracted_sections.json")
output_path = Path("data/textbook--extracted_markdown.md")

with open(input_path, encoding="utf-8") as f:
    sections = json.load(f)

lines = []
for section in sections:
    heading = section["heading"]
    content = section.get("content", "")

    if heading == "Introduction":
        # No heading marker — just write the content
        lines.append(content)
    else:
        # "Section 1.3.1 Foo Bar" -> "## 1.3.1 Foo Bar"
        md_heading = "## " + heading.removeprefix("Section ")
        lines.append(md_heading)
        lines.append(content)

    lines.append("")  # blank line between sections

output_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Written {len(sections)} sections to {output_path}")
