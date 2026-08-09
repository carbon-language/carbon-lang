// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field/fail_unbound_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field/fail_unbound_field.carbon

class Class {
  var field: i32;
  fn F() -> i32 {
    // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+3]]:12-16: error: expression cannot be used as a value [UseOfNonExprAsValue]
    // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE-2]]:13-15: note: return type of function is `i32` [ReturnTypeHereNote]
    // CHECK:STDERR:
    return field;
  }

  fn IsZero() -> bool {
    // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+4]]:18-19: error: type `<unbound element of class Class>` does not implement interface `Core.EqWith(Core.IntLiteral)` [MissingImplForType]
    // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+3]]:12-16: note: left operand has type `<unbound element of class Class>` [BinaryOperatorLhsType]
    // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+2]]:21: note: right operand has type `Core.IntLiteral` [BinaryOperatorRhsType]
    // CHECK:STDERR:
    return field == 0;
  }
}

fn G() -> i32 {
  // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+3]]:10-20: error: expression cannot be used as a value [UseOfNonExprAsValue]
  // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE-2]]:11-13: note: return type of function is `i32` [ReturnTypeHereNote]
  // CHECK:STDERR:
  return Class.field;
}
