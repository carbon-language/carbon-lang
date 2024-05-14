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

fn CallWithReturnedVar() -> C {
  Print("Entering call");
  heap.PrintAllocs();
  returned var c: C = {};
  Print("Object created, returning");
  return var;
}

fn FromInitializingExpression_ReturnedVar() {
  Print("Initialize c from initializing expression (returned var)");
  let c: C = CallWithReturnedVar();
  heap.PrintAllocs();
}

fn Main() -> i32 {
  FromInitializingExpression_ReturnedVar();
  return 0;
}

// CHECK:STDOUT: Initialize c from initializing expression (returned var)
// CHECK:STDOUT: Entering call
// CHECK:STDOUT: 0: Heap{}, 1: !Uninit<class C>
// CHECK:STDOUT: Object created, returning
// CHECK:STDOUT: 0: Heap{}, 1: C{}
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: result: 0
