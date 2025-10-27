// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/require_invalid.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/require_invalid.carbon

// --- fail_todo_require_outside_scope.carbon
library "[[@TEST_NAME]]";

interface Y {}

// CHECK:STDERR: fail_todo_require_outside_scope.carbon:[[@LINE+4]]:1: error: semantics TODO: `require` [SemanticsTodo]
// CHECK:STDERR: require impls Y;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
require impls Y;

// --- fail_todo_require_in_class.carbon
library "[[@TEST_NAME]]";

interface Y {}

class C {
  // TODO: require is not allowed outside of `interface` or `constraint`.
  // CHECK:STDERR: fail_todo_require_in_class.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls Y;
}

// --- fail_require_in_fn.carbon
library "[[@TEST_NAME]]";

interface Y {}

fn F() {
  // require is not allowed outside of `interface` or `constraint`.
  // CHECK:STDERR: fail_require_in_fn.carbon:[[@LINE+8]]:3: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_require_in_fn.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls Y;
}
