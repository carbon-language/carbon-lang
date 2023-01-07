// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Allocate A
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: Delete A
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: result: 0

package ExplorerTest api;


base class A{
    fn Create() -> A {
        return {};
    }
    virtual destructor[self: Self] {
        Print("DESTRUCTOR A");
    }
}

fn Main() -> i32 {
  Print("Allocate A");
  var pa: A* = heap.New(A.Create());

  Print("Delete A");
  heap.Delete(pa);

  return 0;
}
