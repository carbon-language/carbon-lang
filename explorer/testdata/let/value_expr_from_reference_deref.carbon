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

fn FromReferenceExpressionDeref() {
  var c_var: C = {};
  var c_ref: C* = &c_var;
  heap.PrintAllocs();
  Print("Initialize c from reference expression from dereferenced pointer");
  let c: C = *c_ref;
  heap.PrintAllocs();
}

fn Main() -> i32 {
  FromReferenceExpressionDeref();
  return 0;
}

// CHECK:STDOUT: 0: Heap{}, 1: C{}, 2: ptr<Allocation(1)>
// CHECK:STDOUT: Initialize c from reference expression from dereferenced pointer
// CHECK:STDOUT: 0: Heap{}, 1: C{}, 2: ptr<Allocation(1)>
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: result: 0
