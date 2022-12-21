// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Foo(c1): 1
// CHECK:STDOUT: Foo(c2): 1
// CHECK:STDOUT: Foo(d): 1
// CHECK:STDOUT: Foo(&e): 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  var val: i32;
}

base class D extends C {
  var val: i32;
}

class E extends D {
  var val: i32;
}

fn Foo(c: C*) -> i32 {
  return (*c).val;
}

fn Main() -> i32 {
  var e: E = { .val = 3, .base = {.val = 2,.base = {.val = 1}}};
  var d: D* = &e;
  var c1: C* = &e;
  var c2: C* = d;

  Print("Foo(c1): {0}", Foo(c1));
  Print("Foo(c2): {0}", Foo(c2));
  Print("Foo(d): {0}", Foo(d));
  Print("Foo(&e): {0}", Foo(&e));

  return 0;
}
