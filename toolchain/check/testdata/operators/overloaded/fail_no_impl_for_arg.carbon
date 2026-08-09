// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon

package User;

class C {};
class D {};

impl C as Core.AddWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C { return {}; }
}
impl C as Core.AddAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn Test(a: C, b: D, z: bool) -> C {
  if (z) {
    return a + a;
  } else {
    // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+4]]:14: error: type `C` does not implement interface `Core.AddWith(D)` [MissingImplForType]
    // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+3]]:12: note: left operand has type `C` [BinaryOperatorLhsType]
    // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+2]]:16: note: right operand has type `D` [BinaryOperatorRhsType]
    // CHECK:STDERR:
    return a + b;
  }
}

fn TestAssign(b: D) {
  var a: C = {};

  a += a;

  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+4]]:5-6: error: type `C` does not implement interface `Core.AddAssignWith(D)` [MissingImplForType]
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+3]]:3: note: left operand has type `C` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+2]]:8: note: right operand has type `D` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  a += b;
}
