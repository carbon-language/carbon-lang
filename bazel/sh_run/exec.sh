#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Don't pass BUILD_WORKING_DIRECTORY to the subcommand because args will use
# pwd-relative paths.
unset BUILD_WORKING_DIRECTORY
exec "$@"
