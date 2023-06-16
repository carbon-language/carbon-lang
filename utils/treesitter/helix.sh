#!/bin/sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/utils/treesitter"

./build.sh

mkdir -p "$ROOT/.helix"

cat > "$ROOT/.helix/languages.toml" << EOF
use-grammars = { only = ["carbon"] }

[[language]]
name = "carbon"
scope = "source.carbon"
file-types = ["carbon"]
comment-token = "//"
indent = { tab-width = 2, unit = "  " }
roots = [".git"]

[[grammar]]
name = "carbon"
source = { path = "$PWD" }
EOF

mkdir -p ~/.config/helix/runtime/grammars ~/.config/helix/runtime/queries
ln -sTf $PWD/queries ~/.config/helix/runtime/queries/carbon
hx --grammar build

echo
hx --health carbon
echo
echo 'use `hx path/to/foo.carbon` to open files'
echo 'Try different themes with :theme'
