// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: (*c).a: 1
// CHECK:STDOUT: Foo(&d): 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  var a: i32;
}

class D extends C {
}

fn Foo(c: C*) -> i32 {
  return (*c).a;
}

fn Main() -> i32 {
  var d: D = { .base = {.a = 1} };
  var c: C* = &d;
  Print("(*c).a: {0}", (*c).a);
  Print("Foo(&d): {0}", Foo(&d));
  return 0;
}
