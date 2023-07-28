#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# We run the autoupdate scripts serially, but can build them all at once.
bazel build -c opt \
  //toolchain/codegen:codegen_file_test \
  //toolchain/driver:driver_file_test \
  //toolchain/lexer:lexer_file_test \
  //toolchain/lowering:lowering_file_test \
  //toolchain/parser:parse_tree_file_test \
  //toolchain/semantics:semantics_file_test

$(dirname %~dp0)/codegen/autoupdate_testdata.sh
$(dirname %~dp0)/driver/autoupdate_testdata.sh
$(dirname %~dp0)/lexer/autoupdate_testdata.sh
$(dirname %~dp0)/lowering/autoupdate_testdata.sh
$(dirname %~dp0)/parser/autoupdate_testdata.sh
$(dirname %~dp0)/semantics/autoupdate_testdata.sh
