#!/bin/bash -eu
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Removes files added by install.sh.

echo "Uninstalling files using sudo..."
sudo -- /usr/bin/bash - <<EOF
rm -rf /usr/lib/carbon
rm /usr/bin/carbon-cam
EOF
echo "All done."
