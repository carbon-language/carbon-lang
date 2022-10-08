// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: DESTRUCTOR A 1
// CHECK: DESTRUCTOR A 2
// CHECK: DESTRUCTOR A 3
// CHECK: DESTRUCTOR A 1
// CHECK: DESTRUCTOR A 2
// CHECK: DESTRUCTOR A 3
// CHECK: result: 1

package ExplorerTest api;

class A{
    destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    var n: i32;
}

fn Main() -> i32 {
  var i: i32 = 0;
  while( i < 2){
    var a: A = {.n = 3};
    var b: A = {.n = 2};
    var c: A = {.n = 1};
    i = i + 1;
  }
  return 1;
}
