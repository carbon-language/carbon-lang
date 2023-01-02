// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: c.Foo(): 1
// CHECK:STDOUT: d.Foo(): 2
// CHECK:STDOUT: e.Foo(): 3
// CHECK:STDOUT: (*dp).Foo(): 3
// CHECK:STDOUT: (*dc).Foo(): 3
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  var value_c: i32;
  virtual fn Foo[self: Self]() -> i32 {
    return self.value_c;
  }
}

base class D extends C {
  var value_d: i32;
  impl fn Foo[self: Self]() -> i32 {
    return self.value_d;
  }
}

class E extends D {
  var value_e: i32;
  impl fn Foo[self: Self]() -> i32 {
    return self.value_e;
  }
}

fn Main() -> i32 {
  var c: C = {.value_c = 1};
  Print("c.Foo(): {0}", c.Foo());
  var d: D = {.value_d = 2, .base = {.value_c = 1}};
  Print("d.Foo(): {0}", d.Foo());
  var e: E = {.value_e = 3, .base={.value_d = 2, .base = {.value_c = 1}}};
  Print("e.Foo(): {0}", e.Foo());
  var dp: D* = &e;
  Print("(*dp).Foo(): {0}", (*dp).Foo());
  var dc: C* = &e;
  Print("(*dc).Foo(): {0}", (*dc).Foo());

  return 0;
}
