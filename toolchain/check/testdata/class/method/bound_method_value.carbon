// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method/bound_method_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method/bound_method_value.carbon

// --- fail_implicit_conversion.carbon
library "[[@TEST_NAME]]";

class C {
  fn Get(self) -> i32;
}

fn F(c: C) {
  // CHECK:STDERR: fail_implicit_conversion.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `<bound method>` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var _: i32 = c.Get;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_implicit_conversion.carbon:[[@LINE+4]]:3: note: type `<bound method>` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var _: i32 = c.Get;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  var _: i32 = c.Get;
}

// --- fail_as.carbon
library "[[@TEST_NAME]]";

class C {
  fn Get(self) -> i32;
}

fn F(c: C) -> i32 {
  // CHECK:STDERR: fail_as.carbon:[[@LINE+7]]:10: error: cannot convert expression of type `<bound method>` to `i32` with `as` [ConversionFailure]
  // CHECK:STDERR:   return c.Get as i32;
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR: fail_as.carbon:[[@LINE+4]]:10: note: type `<bound method>` does not implement interface `Core.As(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return c.Get as i32;
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR:
  return c.Get as i32;
}

// --- fail_interface_member.carbon
library "[[@TEST_NAME]]";

interface I {
  fn Op(self);
}

class C {
  fn Get(self) -> i32;
}

fn F(c: C) {
  // CHECK:STDERR: fail_interface_member.carbon:[[@LINE+4]]:3: error: cannot access member of interface `I` in type `<bound method>` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   c.Get.(I.Op)();
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  c.Get.(I.Op)();
}

// --- fail_copy.carbon
library "[[@TEST_NAME]]";

class C {
  fn Get(self) -> i32;
}

fn F(c: C) {
  // CHECK:STDERR: fail_copy.carbon:[[@LINE+7]]:26: error: cannot copy value of type `<bound method>` [CopyOfUncopyableType]
  // CHECK:STDERR:   var _: typeof(c.Get) = c.Get;
  // CHECK:STDERR:                          ^~~~~
  // CHECK:STDERR: fail_copy.carbon:[[@LINE+4]]:26: note: type `<bound method>` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var _: typeof(c.Get) = c.Get;
  // CHECK:STDERR:                          ^~~~~
  // CHECK:STDERR:
  var _: typeof(c.Get) = c.Get;
}
