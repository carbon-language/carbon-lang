// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: c.Foo() -> 5
// CHECK:STDOUT: d.Foo() -> 4
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  var a: i32;
  virtual fn Foo[me: Self]() -> i32 {
    return 5;
  }
  fn Bar[me: Self]() -> i32 {
    return 10;
  }
}

class D extends C {
  var b: i32;
  fn Foo[me: Self]() -> i32 {
    return 4;
  }
  fn Bar[me: Self]() -> i32 {
    return 11;
  }
}

fn Main() -> i32 {
  var c: C = {.a = 1};
  Print("c.Foo() -> {0}", c.Foo());
  var d: D = {.b = 3, .base = {.a = 2}};
  Print("d.Foo() -> {0}", d.Foo());
  var cc: C* = &d;
  Print("cc.Foo() -> {0}", (*cc).Foo());
  Print("cc.Bar() -> {0}", (*cc).Bar());
  return 0;
}
