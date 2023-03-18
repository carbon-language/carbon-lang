// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;


base class A{
    virtual destructor[self: Self]{}
}

class B extends A {
    fn Create() -> Self {
        return {.base={}};
    }
    impl destructor[self: Self]{}
}

fn Main() -> i32 {
  var pb: B* = heap.New(B.Create());
  var pa: A* = pb;
  heap.Delete(pa);
  return 0;
}
