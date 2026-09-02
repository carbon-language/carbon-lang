// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/ordered.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/ordered.carbon

// --- user.carbon

package User;

class C {};

impl C as Core.OrderedWith(C) {
  fn Less(self: C, other: C) -> bool;
  fn LessOrEquivalent(self: C, other: C) -> bool;
  fn Greater(self: C, other: C) -> bool;
  fn GreaterOrEquivalent(self: C, other: C) -> bool;
}

fn TestLess(a: C, b: C) -> bool {
  return a < b;
}

fn TestLessEqual(a: C, b: C) -> bool {
  return a <= b;
}

fn TestGreater(a: C, b: C) -> bool {
  return a > b;
}

fn TestGreaterEqual(a: C, b: C) -> bool {
  return a >= b;
}

// --- ordered.carbon

package Ordered;

class C2 {};

impl C2 as Core.Ordered {
  fn Less(self: C2, other: C2) -> bool;
  fn LessOrEquivalent(self: C2, other: C2) -> bool;
  fn Greater(self: C2, other: C2) -> bool;
  fn GreaterOrEquivalent(self: C2, other: C2) -> bool;
}

fn TestLess(a: C2, b: C2) -> bool {
  return a < b;
}

fn TestLessEqual(a: C2, b: C2) -> bool {
  return a <= b;
}

fn TestGreater(a: C2, b: C2) -> bool {
  return a > b;
}

fn TestGreaterEqual(a: C2, b: C2) -> bool {
  return a >= b;
}

// --- fail_no_impl.carbon

package FailNoImpl;

class D {};

fn TestLess(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.OrderedWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a < b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a < b;
}

fn TestLessEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.OrderedWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a <= b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a <= b;
}

fn TestGreater(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.OrderedWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a > b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a > b;
}

fn TestGreaterEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.OrderedWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a >= b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a >= b;
}
