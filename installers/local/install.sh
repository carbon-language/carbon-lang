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
    --install_path)
      INSTALL_PATH="$2"
      shift
      shift
      ;;
    *)
      STANDARD_LIBRARIES+=("$1")
      echo $1
      shift
      ;;
  esac
done

# If the install path is relative, change it to be based on the working dir.
if [[ ! "${INSTALL_PATH}" = /* ]]; then
  INSTALL_PATH="${BUILD_WORKING_DIRECTORY}/${INSTALL_PATH}"
fi

# Prepare the install script to run.
SCRIPT=$(cat <<EOF
  # Ensure directories exist.
  mkdir -p "${INSTALL_PATH}/bin"
  mkdir -p "${INSTALL_PATH}/lib/carbon/data"

  # Install files to lib.
  install -m 755 "${CARBON}" "${INSTALL_PATH}/lib/carbon/carbon"
  for f in $(printf " %q" "${STANDARD_LIBRARIES[@]}"); do
    install -m 644 "\${f}" "${INSTALL_PATH}/lib/carbon/data/"
  done

  # Add symlinks in bin.
  ln -fs "../lib/carbon/carbon" \
    "${INSTALL_PATH}/bin/carbon-explorer"
EOF
)

# Only use sudo if the target directory isn't user-owned.
ACCESS_PATH="${INSTALL_PATH}"
while [[ ! -e "${ACCESS_PATH}" ]]; do
  ACCESS_PATH="$(dirname "${ACCESS_PATH}")"
done

if [[ -O "${ACCESS_PATH}" ]]; then
  echo "Installing files..."
  echo "${SCRIPT}" | /bin/bash -eux -
else
  echo "Installing files using sudo..."
  echo "${SCRIPT}" | sudo -- /bin/bash -eux -
fi
echo "All done."
