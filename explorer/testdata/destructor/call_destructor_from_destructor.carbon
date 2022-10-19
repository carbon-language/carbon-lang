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
// CHECK:STDOUT: result: 1

package ExplorerTest api;

class A{
    destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    fn Create(x: i32) -> A{
        return {.n = x};
    }
    var n: i32;
}

class B{
    destructor[me: Self]{
        Print("DESTRUCTOR B {0}",me.n);
    }
    fn Create(x: i32) -> B{
       return {.n = x, .a = A.Create(3) };
    }
    var n: i32;
    var a: A;
}


fn Main() -> i32 {
  var a: A = A.Create(4);
  var b: B = B.Create(2);
  var c: A = A.Create(1);
  return 1;
}
