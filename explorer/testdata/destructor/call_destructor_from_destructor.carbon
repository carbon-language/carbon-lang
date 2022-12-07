// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: DESTRUCTOR A 1
// CHECK:STDOUT: DESTRUCTOR B 2
// CHECK:STDOUT: DESTRUCTOR A 3
// CHECK:STDOUT: DESTRUCTOR A 4
// CHECK:STDOUT: DESTRUCTOR A 5
// CHECK:STDOUT: result: 1

package ExplorerTest api;

class A{
    destructor[self: Self]{
        Print("DESTRUCTOR A {0}",self.n);
    }
    fn Create(x: i32) -> A{
        return {.n = x};
    }
    var n: i32;
}

class B{
    destructor[self: Self]{
        Print("DESTRUCTOR B {0}",self.n);
    }
    fn Create(x: i32) -> B{
       return {.n = x, .a1 = A.Create(4),.a2 = A.Create(3) };
    }
    var a1: A;
    var n: i32;
    var a2: A;
}


fn Main() -> i32 {
  var a: A = A.Create(5);
  var b: B = B.Create(2);
  var c: A = A.Create(1);
  return 1;
}
