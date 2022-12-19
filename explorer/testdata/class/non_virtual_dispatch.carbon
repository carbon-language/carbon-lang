// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: (*c).a: 1
// CHECK:STDOUT: (*c).Foo(): 1
// CHECK:STDOUT: (*c).Bar(): 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  var a: i32;
  fn Foo[self: Self]() -> i32 {
    return 1;
  }
  fn Bar() -> i32 {
    return 1;
  }
}

class D extends C {
  var b: i32;
  fn Foo[self: Self]() -> i32 {
    return 2;
  }
  fn Bar() -> i32 {
    return 2;
  }
}

fn Main() -> i32 {
  var d: D = { .base = {.a = 1}, .b = 2 };
  var c: C* = &d;
  Print("(*c).a: {0}", (*c).a);
  Print("(*c).Foo(): {0}", (*c).Foo());
  Print("(*c).Bar(): {0}", (*c).Bar());
  return 0;
}
