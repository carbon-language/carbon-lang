#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

ARGS=("$@")
for i in "${!ARGS[@]}"; do
  if [[ -e "${ARGS[$i]}" ]]; then
    ARGS[$i]="$(realpath ${ARGS[$i]})"
  fi
done
exec "${ARGS[@]}"
