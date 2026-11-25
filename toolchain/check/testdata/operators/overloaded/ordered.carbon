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

// --- generic.carbon

package Generic;

class C {};

impl C as Core.OrderedWith(C) {
  fn Less(self: C, other: C) -> bool;
  fn LessOrEquivalent(self: C, other: C) -> bool;
  fn Greater(self: C, other: C) -> bool;
  fn GreaterOrEquivalent(self: C, other: C) -> bool;
}

class C2 {};

impl C2 as Core.Ordered {
  fn Less(self: C2, other: C2) -> bool;
  fn LessOrEquivalent(self: C2, other: C2) -> bool;
  fn Greater(self: C2, other: C2) -> bool;
  fn GreaterOrEquivalent(self: C2, other: C2) -> bool;
}

fn TestGenericLess[T: Core.Ordered](a: T, b: T) -> bool {
  return a < b;
}

fn TestGenericLessEqual[T: Core.Ordered](a: T, b: T) -> bool {
  return a <= b;
}

fn TestGenericGreater[T: Core.Ordered](a: T, b: T) -> bool {
  return a > b;
}

fn TestGenericGreaterEqual[T: Core.Ordered](a: T, b: T) -> bool {
  return a >= b;
}

fn CallGeneric(c: C, c2: C2, i: i32) {
  let _: bool = TestGenericLess(c, c);
  let _: bool = TestGenericLessEqual(c, c);
  let _: bool = TestGenericGreater(c, c);
  let _: bool = TestGenericGreaterEqual(c, c);
  let _: bool = TestGenericLess(c2, c2);
  let _: bool = TestGenericLessEqual(c2, c2);
  let _: bool = TestGenericGreater(c2, c2);
  let _: bool = TestGenericGreaterEqual(c2, c2);
  let _: bool = TestGenericLess(i, i);
  let _: bool = TestGenericLessEqual(i, i);
  let _: bool = TestGenericGreater(i, i);
  let _: bool = TestGenericGreaterEqual(i, i);
}

// --- fail_no_impl.carbon

package FailNoImpl;

class D {};

fn TestLess(a: D, b: D) -> bool {
  // CHECK:STDERR:        ╭─┤ fail_no_impl.carbon:[[@LINE+8]]:10
  // CHECK:STDERR: ->     │   return a < b;
  // CHECK:STDERR:        ·          ┳━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: cannot access member of interface `Core.OrderedWith(D)` in
  // CHECK:STDERR:        ·             ┃        type `D` that does not implement that interface
  // CHECK:STDERR:        ·             ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return a < b;
}

fn TestLessEqual(a: D, b: D) -> bool {
  // CHECK:STDERR:        ╭─┤ fail_no_impl.carbon:[[@LINE+8]]:10
  // CHECK:STDERR: ->     │   return a <= b;
  // CHECK:STDERR:        ·          ┳━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: cannot access member of interface `Core.OrderedWith(D)` in
  // CHECK:STDERR:        ·             ┃        type `D` that does not implement that interface
  // CHECK:STDERR:        ·             ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return a <= b;
}

fn TestGreater(a: D, b: D) -> bool {
  // CHECK:STDERR:        ╭─┤ fail_no_impl.carbon:[[@LINE+8]]:10
  // CHECK:STDERR: ->     │   return a > b;
  // CHECK:STDERR:        ·          ┳━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: cannot access member of interface `Core.OrderedWith(D)` in
  // CHECK:STDERR:        ·             ┃        type `D` that does not implement that interface
  // CHECK:STDERR:        ·             ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return a > b;
}

fn TestGreaterEqual(a: D, b: D) -> bool {
  // CHECK:STDERR:        ╭─┤ fail_no_impl.carbon:[[@LINE+8]]:10
  // CHECK:STDERR: ->     │   return a >= b;
  // CHECK:STDERR:        ·          ┳━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: cannot access member of interface `Core.OrderedWith(D)` in
  // CHECK:STDERR:        ·             ┃        type `D` that does not implement that interface
  // CHECK:STDERR:        ·             ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return a >= b;
}
