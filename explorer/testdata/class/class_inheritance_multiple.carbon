// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class A {
  fn FunctionA() {}
  var var_a: i32;
}

base class B extends A {
  fn FunctionB() {}
  var var_b: i32;
}

class C extends B {
  fn FunctionC() {}
  var var_c: i32;
}

fn Main() -> i32 {
  var c: C = {.var_a=1, .var_b=2, .var_c=3};
  c.FunctionA();
  c.FunctionB();
  c.FunctionC();
  return 0;
}
