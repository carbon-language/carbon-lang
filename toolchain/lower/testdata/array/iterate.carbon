// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/iterate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/iterate.carbon

fn G(n: i32);

fn F() {
  // CHECK:STDERR: iterate.carbon:[[@LINE+4]]:26: error: semantics TODO: `BISH: init_inst: {kind: Converted, arg0: inst7E00004D, arg1: inst7E0002AA, type: type(inst7E000039)}` [SemanticsTodo]
  // CHECK:STDERR:   let a: array(i32, 6) = (1, 2, 3, 4, 5, 6);
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: array(i32, 6) = (1, 2, 3, 4, 5, 6);
  for (n: i32 in a) {
    G(n);
  }
}
