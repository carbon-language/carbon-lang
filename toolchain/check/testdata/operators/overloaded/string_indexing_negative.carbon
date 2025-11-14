// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/string_indexing_negative.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/string_indexing_negative.carbon


// --- fail_negative_index.carbon

fn TestNegativeIndex() {
  let test_str: str = "Test";
  // CHECK:STDERR: fail_negative_index.carbon:[[@LINE+4]]:31: error: index `-1` is negative. [ArrayIndexNegative]
  // CHECK:STDERR:   let c: Core.Char = test_str[-1];
  // CHECK:STDERR:                               ^~
  // CHECK:STDERR:
  let c: Core.Char = test_str[-1];
}
