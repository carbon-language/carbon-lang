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
    Print("A destructor");
  }
  var data: i32;
}

fn returned_var() -> i32 {
  returned var c: i32 = 0;
  c = 3;
  return var;
}

fn returned_expr() -> i32 {
  var c: i32 = 5;
  return c;
}

fn returned_expr_a() -> A {
  var a: A = {.data = 6};
  return a;
}

fn no_returned_value() {
  return;
}

fn Main() -> i32 {
  // var x: i32 = returned_var();
  // Print("x = {}", x);
  // Print("====================================");
  // heap.PrintAllocations();
  // Print("====================================");
  var y: i32 = returned_expr();
  Print("y = {0}", y);

  var a: A = returned_expr_a();
  Print("a = {0}", a.data);

  Print("====================================");
  heap.PrintAllocations();
  Print("====================================");
  // no_returned_value();

  // heap.PrintAllocations();

  // let y: i32 = f();
  // Print("y = {}", y);
  return 0;
}
