#!/bin/bash -eu
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Performs a local install. This will replace any previous carbon installs
# without prompting.

STANDARD_LIBRARIES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --carbon)
      CARBON="$2"
      shift
      shift
      ;;
    *)
      STANDARD_LIBRARIES+=("$1")
      shift
      ;;
  esac
done

echo "Installing files using sudo..."
sudo -- /usr/bin/bash -eu - <<EOF
mkdir -p /usr/lib/carbon/data
cp -f "${CARBON}" /usr/lib/carbon/carbon
chmod 755 /usr/lib/carbon/carbon
ln -fs /usr/lib/carbon/carbon /usr/bin/carbon-cam
for f in "${STANDARD_LIBRARIES}"; do
  cp -f "\${f}" /usr/lib/carbon/data/
  chmod 644 "/usr/lib/carbon/data/\$(basename "\${f}")"
done
EOF
echo "All done."
