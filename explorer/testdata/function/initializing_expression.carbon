// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A {
  destructor[self: Self] {
    Print("a{0} destroyed", self.data);
  }
  var data: i32;
}

fn returned_expr() -> i32 {
  var c: i32 = 1;
  return c;
}

fn returned_expr_class() -> A {
  var a: A = {.data = 0};
  a.data = 1;
  return a;
}

fn no_returned_value() {
  return;
}

fn returned_var() -> i32 {
  returned var c: i32 = 0;
  c = 3;
  return var;
}

fn returned_var_class() -> A {
  returned var a: A = {.data = 1};
  a.data = 2;
  return var;
}

fn Main() -> i32 {
  var n1: i32 = returned_expr();
  Print("n1 = {0}", n1);

  var a1: A = returned_expr_class();
  Print("a1.data = {0}", a1.data);

  no_returned_value();

  var n2: i32 = returned_var();
  Print("n2 = {0}", n2);

  var a2: A = returned_var_class();
  Print("a2.data = {0}", a2.data);

  heap.PrintAllocations();
  // TODO: let

  return 0;
}
