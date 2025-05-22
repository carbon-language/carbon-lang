// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/self_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/self_type.carbon

// --- self_type.carbon

library "[[@TEST_NAME]]";

class Class {
  fn F[self: Self]() -> ();
  fn Make() -> Self {
    returned var s: Self;
    s = {.p = &s};
    return var;
  }
  var p: Self*;
}

fn Class.F[self: Self]() -> () {
  return (*self.p).F();
}

// --- self_conversion.carbon

library "[[@TEST_NAME]]";

base class Base {
  var a: ();
}

class Derived {
  extend base: Base;

  fn SelfBase[self: Base]() -> ();
  fn AddrSelfBase[addr self: Base*]();
}

fn Derived.SelfBase[self: Base]() -> () {
  return self.a;
}

fn Derived.AddrSelfBase[addr self: Base*]() {
  (*self).a = ();
}

fn Call(p: Derived*) -> () {
  (*p).AddrSelfBase();
  return (*p).SelfBase();
}

// --- fail_self_type_member.carbon

class Class {
  // CHECK:STDERR: fail_self_type_member.carbon:[[@LINE+4]]:10: error: name `Core.Bool` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   var b: bool;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
  var b: bool;
}

// CHECK:STDERR: fail_self_type_member.carbon:[[@LINE+4]]:11: error: name `Core.Bool` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn F() -> bool {
// CHECK:STDERR:           ^~~~
// CHECK:STDERR:
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
