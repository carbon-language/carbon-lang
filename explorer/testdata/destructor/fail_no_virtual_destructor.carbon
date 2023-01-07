// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;


base class A{
    destructor[self: Self]{
        Print("DESTRUCTOR A");
    }
}

class B extends A {
    destructor[self: Self]{
        Print("DESTRUCTOR B");
    }
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/destructor/fail_no_virtual_destructor.carbon:[[@LINE+1]]: type error in name binding: 'type*' is not implicitly convertible to 'class B*'
  var pb: B* = heap.New(B);
  var pa: A* = pb;
  heap.Delete(pa);
  return 0;
}
