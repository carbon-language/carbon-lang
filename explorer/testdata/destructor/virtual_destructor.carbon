// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Allocate C
// CHECK:STDOUT: DESTRUCTOR C
// CHECK:STDOUT: DESTRUCTOR B
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: Delete C from A*
// CHECK:STDOUT: DESTRUCTOR C
// CHECK:STDOUT: DESTRUCTOR B
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: result: 0

package ExplorerTest api;


base class A {
    virtual destructor[self: Self] {
        Print("DESTRUCTOR A");
    }
}

base class B extends A {
    impl destructor[self: Self] {
        Print("DESTRUCTOR B");
    }
}

class C extends B {
    fn Create() -> Self{
        return {.base={.base={}}};
    }
    impl destructor[self: Self] {
        Print("DESTRUCTOR C");
    }
}

fn Main() -> i32 {
  Print("Allocate C");
  var pc: C* = heap.New(C.Create());
  var pa: A* = pc;

  Print("Delete C from A*");
  heap.Delete(pa);

  return 0;
}
