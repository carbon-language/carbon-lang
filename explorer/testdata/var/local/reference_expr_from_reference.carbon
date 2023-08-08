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

fn FromReferenceExpression() {
  var c_var: C = {};
  heap.PrintAllocs();
  Print("Initialize c from reference expression");
  var c: C = c_var;
  heap.PrintAllocs();
}

fn Main() -> i32 {
  FromReferenceExpression();
  return 0;
}

// CHECK:STDOUT: 0: Heap{}, 1: C{}
// CHECK:STDOUT: Initialize c from reference expression
// CHECK:STDOUT: 0: Heap{}, 1: C{}, 2: C{}
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: result: 0
