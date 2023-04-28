// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: (*c1).val: 1
// CHECK:STDOUT: (*c2).val: 1
// CHECK:STDOUT: (*d).val: 2
// CHECK:STDOUT: e.val: 3
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
  Print("(*c1).val: {0}", (*c1).val);
  Print("(*c2).val: {0}", (*c2).val);
  Print("(*d).val: {0}", (*d).val);
  Print("e.val: {0}", e.val);

  return 0;
}
