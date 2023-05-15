// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: RUNTIME ERROR: explorer/data/prelude.carbon:{{.*}}: Deallocating a derived class from base class pointer requires a virtual destructor

package ExplorerTest api;


base class A{
    destructor[self: Self]{}
}

class B extends A {
    fn Create() -> Self {
        return {.base={}};
    }
    destructor[self: Self]{}
}

fn Main() -> i32 {
  var pb: B* = heap.New(B.Create());
  var pa: A* = pb;
  heap.Delete(pa);
  return 0;
}
