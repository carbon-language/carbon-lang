// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Allocate B
// CHECK:STDOUT: DESTRUCTOR B
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: Delete B from A*
// CHECK:STDOUT: DESTRUCTOR B
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: result: 0

package ExplorerTest api;


base class A{
    virtual destructor[self: Self] {
        Print("DESTRUCTOR A");
    }
}

class B extends A {
    fn Create() -> Self{
        return {.base={}};
    }
    impl destructor[self: Self] {
        Print("DESTRUCTOR B");
    }
}

fn Main() -> i32 {
  Print("Allocate B");
  var pb: B* = heap.New(B.Create());
  var pa: A* = pb;

  Print("Delete B from A*");
  heap.Delete(pa);

  return 0;
}
