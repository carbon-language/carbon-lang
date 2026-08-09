// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/eq.carbon

// --- user.carbon

package User;

class C {};

impl C as Core.EqWith(C) {
  fn Equal(self: C, other: C) -> bool;
  fn NotEqual(self: C, other: C) -> bool;
}

fn TestEqual(a: C, b: C) -> bool {
  return a == b;
}

fn TestNotEqual(a: C, b: C) -> bool {
  return a != b;
}

// --- eq_and_default.carbon

package EqAndDefault;

class C2 {};

impl C2 as Core.Eq {
  fn Equal(self: C2, other: C2) -> bool;
}

fn TestEqual(a: C2, b: C2) -> bool {
  return a == b;
}

fn TestNotEqual(a: C2, b: C2) -> bool {
  return a != b;
}

// --- fail_no_impl.carbon

package FailNoImpl;

class D {};

fn TestEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:12-13: error: type `D` does not implement interface `Core.EqWith(D)` [MissingImplForType]
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:10: note: left operand has type `D` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+2]]:15: note: right operand has type `D` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  return a == b;
}

fn TestNotEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:12-13: error: type `D` does not implement interface `Core.EqWith(D)` [MissingImplForType]
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:10: note: left operand has type `D` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+2]]:15: note: right operand has type `D` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  return a != b;
}

// --- fail_no_impl_for_args.carbon

package FailNoImplForArgs;

class C {};
class D {};

impl C as Core.EqWith(C) {
  fn Equal(self: C, other: C) -> bool;
  fn NotEqual(self: C, other: C) -> bool;
}

fn TestRhsBad(a: C, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+4]]:12-13: error: type `C` does not implement interface `Core.EqWith(D)` [MissingImplForType]
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+3]]:10: note: left operand has type `C` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+2]]:15: note: right operand has type `D` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  return a == b;
}

fn TestLhsBad(a: D, b: C) -> bool {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+4]]:12-13: error: type `D` does not implement interface `Core.EqWith(C)` [MissingImplForType]
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+3]]:10: note: left operand has type `D` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+2]]:15: note: right operand has type `C` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  return a != b;
}
