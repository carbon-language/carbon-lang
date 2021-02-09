#!/usr/bin/bash
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e -u -o pipefail

MYSRCDIR="${TEST_SRCDIR}"
GOLDEN=$1
SUBJECT=$2

if [[ $# == 3 && $3 == "--update" ]]; then
  cp "${SUBJECT}" "${GOLDEN}"
  exit $?
fi

CMD=("diff" "-u" "${GOLDEN}" "${SUBJECT}")

if "${CMD[@]}"; then
  echo "PASS"
  exit 0
fi

cat <<EOT
When running under:
  ${MYSRCDIR}
the golden contents of:
  ${GOLDEN}
do not match generated target:
  ${SUBJECT}

To update the golden file, run the following:

  bazel run //${TEST_BINARY%/*}:${TEST_BINARY##*/} -- --update
EOT

exit 1
