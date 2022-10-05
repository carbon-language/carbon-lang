// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: TEST
// CHECK: TEST 2
// CHECK: DESTRUCTOR A 1
// CHECK: result: 1

package ExplorerTest api;

class B{
    var n : i32;
}

class A{
    destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    fn test[me: Self](){
        Print("TEST");
    }
    fn test2[me: Self](){
        Print("TEST 2");
    }
    var n: i32;
}

fn Main() -> i32 {
  var a: A = {.n = 1 };
  a.test();
  a.test2();
  return 1;
}
