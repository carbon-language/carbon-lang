#!/bin/bash -eu
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Removes files added by install.sh.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install_path)
      INSTALL_PATH="$2"
      shift
      shift
      ;;
    *)
      echo "Unexpected argument: $1"
      exit 1
      ;;
  esac
done

# If the install path is relative, change it to be based on the working dir.
if [[ ! "${INSTALL_PATH}" = /* ]]; then
  INSTALL_PATH="${BUILD_WORKING_DIRECTORY}/${INSTALL_PATH}"
fi

# Prepare the uninstall script to run.
# TODO: As more files are added, consider sharing better with install.sh. Maybe
# still keep deleting legacy (no longer installed) files.
SCRIPT=$(cat <<EOF
  # Clean up deliberately installed files.
  rm -f "${INSTALL_PATH}/bin/carbon-explorer"
  rm -rf "${INSTALL_PATH}/lib/carbon"

  # Clean up higher level directories in case we created them.
  rmdir -p "${INSTALL_PATH}/bin" || true
  rmdir -p "${INSTALL_PATH}/lib" || true
EOF
)

# Only use sudo if the target directory isn't user-owned.
ACCESS_PATH="${INSTALL_PATH}"
while [[ ! -e "${ACCESS_PATH}" ]]; do
  ACCESS_PATH="$(dirname "${ACCESS_PATH}")"
done

if [[ -O "${ACCESS_PATH}" ]]; then
  echo "Uninstalling files..."
  echo "${SCRIPT}" | /usr/bin/bash -eux -
else
  echo "Uninstalling files using sudo..."
  echo "${SCRIPT}" | sudo -- /usr/bin/bash -eux -
fi
echo "All done."
