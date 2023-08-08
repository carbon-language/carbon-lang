// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

base class A {
  fn FunctionA() {}
  var a: i32;
  var aa: String;
}

base class B {
  extend base: A;
  fn FunctionB() {}
  var b: i32;
}

class C {
  extend base: B;
  fn FunctionC() {}
  var c: i32;
}

fn Main() -> i32 {
  var c: C = {.base={.base={.aa="A", .a=1}, .b=2, }, .c=3};
  c.FunctionA();
  c.FunctionB();
  c.FunctionC();
  Print(c.aa);
  Print("c.a: {0}", c.a);
  Print("c.b: {0}", c.b);
  Print("c.c: {0}", c.c);
  return 0;
}

// CHECK:STDOUT: A
// CHECK:STDOUT: c.a: 1
// CHECK:STDOUT: c.b: 2
// CHECK:STDOUT: c.c: 3
// CHECK:STDOUT: result: 0
