// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/unexpected_runtime.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/unexpected_runtime.carbon

// --- fail_unexpected_runtime.carbon

var x: type;

interface Z {
  let T:! type;
}

class D {
  // CHECK:STDERR: fail_unexpected_runtime.carbon:[[@LINE+4]]:31: error: expression is runtime; expected constant [EvalRequiresConstantValue]
  // CHECK:STDERR:   extend impl as Z where .T = x {}
  // CHECK:STDERR:                               ^
  // CHECK:STDERR:
  extend impl as Z where .T = x {}
}
