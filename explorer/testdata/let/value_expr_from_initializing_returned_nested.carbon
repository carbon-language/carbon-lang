// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  destructor[self: Self] {
    Print("c destroyed");
  }
}

fn CallWithReturnedVar2() -> C {
  Print("Before nested init");
  heap.PrintAllocs();
  returned var c: C = {};
  Print("Nested call return");
  heap.PrintAllocs();
  return var;
}

fn CallWithReturnedVar() -> C {
  returned var c: C = CallWithReturnedVar2();
  Print("First call return");
  heap.PrintAllocs();
  return var;
}

fn FromInitializingExpression_ReturnedVar() {
  Print("Initialize c1 from initializing expression (returned var)");
  let c: C = CallWithReturnedVar();
  Print("Declaration scope");
  heap.PrintAllocs();
}

fn Main() -> i32 {
  FromInitializingExpression_ReturnedVar();
  return 0;
}

// CHECK:STDOUT: Initialize c1 from initializing expression (returned var)
// CHECK:STDOUT: Before nested init
// CHECK:STDOUT: 0: Heap{}, 1: !Uninit<class C>
// CHECK:STDOUT: Nested call return
// CHECK:STDOUT: 0: Heap{}, 1: C{}
// CHECK:STDOUT: First call return
// CHECK:STDOUT: 0: Heap{}, 1: C{}
// CHECK:STDOUT: Declaration scope
// CHECK:STDOUT: 0: Heap{}, 1: C{}
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: result: 0
