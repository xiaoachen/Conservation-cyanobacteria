#!/usr/bin/env bash
# Repoint every hard-coded analysis-tree path in this release at a directory of your choice.
# Usage: bash tools/set_project_root.sh /path/to/your/analysis/tree
set -euo pipefail
OLD="/home/yangyicheng/27.evolution"
NEW="${1:?usage: bash tools/set_project_root.sh /path/to/analysis/tree}"
NEW="${NEW%/}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
n=0
while IFS= read -r f; do
    if grep -q "$OLD" "$f"; then
        sed -i.bak "s|$OLD|$NEW|g" "$f" && rm -f "$f.bak"
        echo "  rewritten: ${f#$HERE/}"
        n=$((n+1))
    fi
done < <(find "$HERE" \( -name '*.py' -o -name '*.sh' \) -not -path '*/.git/*')
echo "$n files rewritten: $OLD -> $NEW"
echo "PATHS.md lists every line that was changed."
