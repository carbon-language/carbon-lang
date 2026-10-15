// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/variadics/pack_expansion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/variadics/pack_expansion.carbon

// --- fail_todo_pack_expansion_statement.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_todo_pack_expansion_statement.carbon:[[@LINE+4]]:3: error: semantics TODO: `HandlePackExpansionStart` [SemanticsTodo]
  // CHECK:STDERR:   ... F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  ... F();
}

// --- fail_todo_expand.carbon
library "[[@TEST_NAME]]";

fn F(x: (i32, i32)) {
  // CHECK:STDERR: fail_todo_expand.carbon:[[@LINE+4]]:3: error: semantics TODO: `HandlePrefixOperatorExpand` [SemanticsTodo]
  // CHECK:STDERR:   expand x;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  expand x;
}
