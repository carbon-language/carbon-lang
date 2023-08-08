// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class D {
  destructor[self: Self] {
    Print("d destroyed");
  }
}

class C {
  destructor[self: Self] {
    Print("c destroyed");
  }
  var d: D;
}

fn FromReferenceExpressionDeref() {
  var c_var: C = {.d = {}};
  heap.PrintAllocs();
  Print("Initialize d from reference expression from subobject");
  let d: D = c_var.d;
  heap.PrintAllocs();
}

fn Main() -> i32 {
  FromReferenceExpressionDeref();
  return 0;
}

// CHECK:STDOUT: 0: Heap{}, 1: C{.d = D{}}
// CHECK:STDOUT: Initialize d from reference expression from subobject
// CHECK:STDOUT: 0: Heap{}, 1: C{.d = D{}}
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: d destroyed
// CHECK:STDOUT: result: 0
