// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: DESTRUCTOR A 0
// CHECK:STDOUT: DESTRUCTOR A 1
// CHECK:STDOUT: DESTRUCTOR A 2
// CHECK:STDOUT: result: 1

package ExplorerTest api;


class A{
    destructor[self: Self]{
        Print("DESTRUCTOR A {0}",self.n);
    }
    var n: i32;
}

fn Main() -> i32 {
  var a1: A = {.n = 2};
  var a: [A; 2] = ({.n = 1},{.n = 0});
  return 1;
}
