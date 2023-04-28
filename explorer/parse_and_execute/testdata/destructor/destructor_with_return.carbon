// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: DESTRUCTOR A 3
// CHECK:STDOUT: result: 1

package ExplorerTest api;

class A{
    destructor[self: Self]{
        Print("DESTRUCTOR A {0}",self.n);
        if(self.n == 3){
            return;
        }
        Print("DESTRUCTOR A: This message should not printed");
    }
    var n: i32;
}

fn Main() -> i32 {
  var a: A = {.n = 3};
  return 1;
}
