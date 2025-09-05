// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_self_type_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_self_type_member.carbon

class Class {
  var b: bool;
}

fn F() -> bool {
  var c1: Class = {.b = true};
  // CHECK:STDERR: fail_self_type_member.carbon:[[@LINE+8]]:17: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:   var c2: Class.Self = c1;
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_type_member.carbon:[[@LINE+4]]:17: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var c2: Class.Self = c1;
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  var c2: Class.Self = c1;
  return c2.b;
}
