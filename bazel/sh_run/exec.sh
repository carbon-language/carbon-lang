#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Turn any pwd-relative files into absolute paths. Other args may be
# BUILD_WORKING_DIRECTORY-relative, which will be handled by the executed
# binary.
ARGS=("$@")
for i in "${!ARGS[@]}"; do
  if [[ -e "${ARGS[$i]}" ]]; then
    ARGS[$i]="$(realpath ${ARGS[$i]})"
  fi
done
exec "${ARGS[@]}"
