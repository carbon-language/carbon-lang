// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:DESTRUCTOR A 3
// CHECK:result: 1

package ExplorerTest api;

class A{
    destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
        if(me.n == 3){
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
