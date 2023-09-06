// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Initialize c from initializing expression (return <expr>)
// CHECK:STDOUT: 0: Heap{}, 1: !Uninit<class C>, 2: C{}
// CHECK:STDOUT: Object created, returning
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: 0: Heap{}, 1: C{}, 2: !!C{}
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class C {
  destructor[self: Self] {
    Print("c destroyed");
  }
}

fn CallWithReturnExpression() -> C {
  var c: C = {};
  heap.PrintAllocs();
  Print("Object created, returning");
  return c;
}

fn FromInitializingExpression_ReturnExpr() {
  Print("Initialize c from initializing expression (return <expr>)");
  let c: C = CallWithReturnExpression();
  heap.PrintAllocs();
}

fn Main() -> i32 {
  FromInitializingExpression_ReturnExpr();
  return 0;
}
