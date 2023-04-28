// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: c.Foo() -> 1
// CHECK:STDOUT: c.Bar() -> 2
// CHECK:STDOUT: d.Foo() -> 3
// CHECK:STDOUT: d.Bar() -> 4
// CHECK:STDOUT: cc.Foo() -> 3
// CHECK:STDOUT: cc.Bar() -> 2
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  virtual fn Foo[self: Self]() -> i32 {
    return 1;
  }
  fn Bar[self: Self]() -> i32 {
    return 2;
  }
}

class D extends C {
  impl fn Foo[self: Self]() -> i32 {
    return 3;
  }
  fn Bar[self: Self]() -> i32 {
    return 4;
  }
}

fn Main() -> i32 {
  var c: C = {};
  Print("c.Foo() -> {0}", c.Foo());
  Print("c.Bar() -> {0}", c.Bar());
  var d: D = {.base = {}};
  Print("d.Foo() -> {0}", d.Foo());
  Print("d.Bar() -> {0}", d.Bar());
  var cc: C* = &d;
  Print("cc.Foo() -> {0}", (*cc).Foo());
  Print("cc.Bar() -> {0}", (*cc).Bar());
  return 0;
}
