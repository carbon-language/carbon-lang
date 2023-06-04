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
  Print("No returned value");
  no_returned_value();
  heap.PrintAllocations();

  // Reference expressions from Initializing expression
  Print("Reference expressions from Initializing expression");
  var n1: i32 = returned_expr();
  Print("n1 = {0}", n1);

  var a1: A = returned_expr_class();
  Print("a1.data = {0}", a1.data);

  var n2: i32 = returned_var();
  Print("n2 = {0}", n2);

  var a2: A = returned_var_class();
  Print("a2.data = {0}", a2.data);

  heap.PrintAllocations();

  // Value expressions from Initializing expression
  Print("Value expressions from Initializing expression");

  let ln1: i32 = returned_expr();
  Print("ln1 = {0}", ln1);

  let la1: A = returned_expr_class();
  Print("la1.data = {0}", la1.data);

  let ln2: i32 = returned_var();
  Print("ln2 = {0}", ln2);

  let la2: A = returned_var_class();
  Print("la2.data = {0}", la2.data);

  heap.PrintAllocations();
  return 0;
}
