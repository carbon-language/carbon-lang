// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Allocate D
// CHECK:STDOUT: DESTRUCTOR B
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: DESTRUCTOR D
// CHECK:STDOUT: DESTRUCTOR C
// CHECK:STDOUT: Delete B from A*
// CHECK:STDOUT: DESTRUCTOR B
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: Delete D from C*
// CHECK:STDOUT: DESTRUCTOR D
// CHECK:STDOUT: DESTRUCTOR C
// CHECK:STDOUT: result: 0

package ExplorerTest api;


base class A {
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

base class C {
    virtual destructor[self: Self] {
        Print("DESTRUCTOR C");
    }
}

class D extends C {
    fn Create() -> Self{
        return {.base={}, .d_pa=heap.New(B.Create())};
    }
    impl destructor[self: Self] {
        Print("DESTRUCTOR D");
    }
    var d_pa: A*;
}

fn Main() -> i32 {
  Print("Allocate D");
  var pd: D* = heap.New(D.Create());
  var pc: C* = pd;

  Print("Delete B from A*");
  heap.Delete(pd->d_pa);

  Print("Delete D from C*");
  heap.Delete(pc);

  return 0;
}
