// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_unbound_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_unbound_field.carbon

class Class {
  var field: i32;
  fn F() -> i32 {
    // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+4]]:12: error: expression cannot be used as a value [UseOfNonExprAsValue]
    // CHECK:STDERR:     return field;
    // CHECK:STDERR:            ^~~~~
    // CHECK:STDERR:
    return field;
  }
}

fn G() -> i32 {
  // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+4]]:10: error: expression cannot be used as a value [UseOfNonExprAsValue]
  // CHECK:STDERR:   return Class.field;
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR:
  return Class.field;
}
