// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Allocate B
// CHECK:STDOUT: DESTRUCTOR B
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: Delete B
// CHECK:STDOUT: DESTRUCTOR B
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: Return
// CHECK:STDOUT: result: 0

package ExplorerTest api;


base class A{
    destructor[self: Self] {
        Print("DESTRUCTOR A");
    }
}

class B extends A {
    fn Create() -> Self{
        return {.base={}};
    }
    destructor[self: Self] {
        Print("DESTRUCTOR B");
    }
}

fn Main() -> i32 {
  Print("Allocate B");
  var pb: B* = heap.New(B.Create());

  Print("Delete B");
  heap.Delete(pb);

  Print("Return");
  return 0;
}
