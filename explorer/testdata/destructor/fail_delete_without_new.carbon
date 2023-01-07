// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}
// CHECK:STDOUT: DESTRUCTOR B
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: result: 0

package ExplorerTest api;


base class A{
    virtual destructor[self: Self]{
        Print("DESTRUCTOR A");
    }
}

class B extends A {
    impl destructor[self: Self]{
        Print("DESTRUCTOR B");
    }
}

fn Main() -> i32 {
  var b: B = {.base={}};
  var pb: B* = &b;
  heap.Delete(pb);
  return 0;
}
