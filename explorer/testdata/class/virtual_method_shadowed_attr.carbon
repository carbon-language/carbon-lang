// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: c.Foo(): 1
// CHECK:STDOUT: d.Foo(): 2
// CHECK:STDOUT: (*cp).Foo(): 2
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  var value: i32;
  virtual fn Foo[self: Self]() -> i32 {
    return self.value;
  }
}

base class D extends C {
  var value: i32;
  impl fn Foo[self: Self]() -> i32 {
    return self.value;
  }
}

fn Main() -> i32 {
  var c: C = {.value = 1};
  Print("c.Foo(): {0}", c.Foo());
  var d: D = {.value = 2, .base = {.value = 1}};
  Print("d.Foo(): {0}", d.Foo());
  var cp: C* = &d;
  Print("(*cp).Foo(): {0}", (*cp).Foo());

  return 0;
}
