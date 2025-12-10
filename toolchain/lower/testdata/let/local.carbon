// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/let/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/let/local.carbon

// --- local.carbon
library "[[@TEST_NAME]]";

fn Run() -> i32 {
  let n: i32 = 1;
  let m: i32 = n;
  return m;
}

// --- symbolic.carbon
library "[[@TEST_NAME]]";

class C {}

fn F(c: C) {}

fn CallF() {
  // CHECK:STDERR: symbolic.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let c:! C = {};
  // CHECK:STDERR:       ^~~~~
  // CHECK:STDERR:
  let c:! C = {};
  F(c);
}
