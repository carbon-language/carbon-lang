// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/virtual.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/virtual.carbon

// --- classes.carbon

package Classes;

base class Base {
}

base class Intermediate {
  extend base: Base;
  virtual fn Fn[self: Self]() { }
}

class Derived {
  extend base: Intermediate;
  impl fn Fn[self: Self]() { }
}

// --- create.carbon

package Create;

import Classes;

fn Create() {
  var b: Classes.Base = {};
  var i: Classes.Intermediate = {.base = {}};
  var d: Classes.Derived = {.base = {.base = {}}};
  // TODO: Support vptr initialization without explicit source initializers.
  // CHECK:STDERR: create.carbon:[[@LINE+8]]:10: error: expected `:` or `:!` in binding pattern [ExpectedBindingPattern]
  // CHECK:STDERR:   var d2 = Classes.Derived;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  // CHECK:STDERR: create.carbon:[[@LINE+4]]:10: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var d2 = Classes.Derived;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  var d2 = Classes.Derived;
}

fn Use(v: Classes.Intermediate*) {
  v->Fn();
}

// --- member_init.carbon

package MemberInit;

base class Base {
  var m: i32;
  virtual fn Fn[self: Self]() { }
}

fn Fn() {
  var i: i32 = 3;
  var v: Base = {.m = i};
  v.m = 5;
  var u: Base = {.m = 3};
}

// --- member_brace_init.carbon

library "[[@TEST_NAME]]";

base class Base {
  virtual fn F[self: Self]();
}

class Derived {
  extend base: Base;
}

fn Use() {
  var v : Derived = {.base = {}};
}
