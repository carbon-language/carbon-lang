// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: TEST
// CHECK:STDOUT: TEST 2
// CHECK:STDOUT: DESTRUCTOR A 1
// CHECK:STDOUT: result: 1

package ExplorerTest api;

class B{
    var n : i32;
}

class A{
    destructor[self: Self]{
        Print("DESTRUCTOR A {0}",self.n);
    }
    fn test[self: Self](){
        Print("TEST");
    }
    fn test2[self: Self](){
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
