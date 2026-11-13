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
  // CHECK:STDERR:        ╭─┤ fail_implicit_conversion.carbon:[[@LINE+11]]:3
  // CHECK:STDERR: ->     │   var _: i32 = c.Get;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert expression of type `<bound method>` to
  // CHECK:STDERR:        ·   │  ┃        `i32`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·   ╰────┤ type `<bound method>` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.ImplicitAs(i32)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var _: i32 = c.Get;
}

// --- fail_as.carbon
library "[[@TEST_NAME]]";

class C {
  fn Get(self) -> i32;
}

fn F(c: C) -> i32 {
  // CHECK:STDERR:        ╭─┤ fail_as.carbon:[[@LINE+11]]:10
  // CHECK:STDERR: ->     │   return c.Get as i32;
  // CHECK:STDERR:        ·          ┳━━━━━━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: cannot convert expression of type `<bound method>` to `i32`
  // CHECK:STDERR:        ·          │  ┃        with `as`
  // CHECK:STDERR:        ·          │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·          ╰────┤ type `<bound method>` does not implement interface
  // CHECK:STDERR:        ·               │ `Core.As(i32)`
  // CHECK:STDERR:        ·               │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_interface_member.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   c.Get.(I.Op)();
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot access member of interface `I` in type `<bound method>`
  // CHECK:STDERR:        ·      ┃        that does not implement that interface
  // CHECK:STDERR:        ·      ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  c.Get.(I.Op)();
}

// --- fail_copy.carbon
library "[[@TEST_NAME]]";

class C {
  fn Get(self) -> i32;
}

fn F(c: C) {
  // CHECK:STDERR:        ╭─┤ fail_copy.carbon:[[@LINE+10]]:26
  // CHECK:STDERR: ->     │   var _: typeof(c.Get) = c.Get;
  // CHECK:STDERR:        ·                          ┳━━━━
  // CHECK:STDERR:        ·                          ┗━━┫ error: cannot copy value of type `<bound method>`
  // CHECK:STDERR:        ·                          │  ┃        [CopyOfUncopyableType]
  // CHECK:STDERR:        ·                          ╰────┤ type `<bound method>` does not implement
  // CHECK:STDERR:        ·                               │ interface `Core.Copy`
  // CHECK:STDERR:        ·                               │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var _: typeof(c.Get) = c.Get;
}
