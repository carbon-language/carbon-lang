// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: DESTRUCTOR A 1
// CHECK:STDOUT: DESTRUCTOR B 2
// CHECK:STDOUT: result: 1

package ExplorerTest api;

class A{
    class B{
        destructor[self: Self]{
            Print("DESTRUCTOR B {0}",self.n);
        }
        fn Create(x: i32) -> B{
           return {.n = x };
        }
        var n: i32;
    }

    destructor[self: Self]{
        Print("DESTRUCTOR A {0}",self.n);
    }
    fn Create(x: i32) -> A{
        return {.n = x, .b = B.Create(2)};
    }
    var n: i32;
    var b : B;
}

fn Main() -> i32 {
  var a: A = A.Create(1);
  return 1;
}
