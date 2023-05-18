// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Allocate A
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: Delete A
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: Return
// CHECK:STDOUT: result: 0

package ExplorerTest api;


class A{
    fn Create() -> A {
        return {};
    }
    destructor[self: Self] {
        Print("DESTRUCTOR A");
    }
}

fn Main() -> i32 {
  Print("Allocate A");
  var pa: A* = heap.New(A.Create());

  Print("Delete A");
  heap.Delete(pa);

  Print("Return");
  return 0;
}
