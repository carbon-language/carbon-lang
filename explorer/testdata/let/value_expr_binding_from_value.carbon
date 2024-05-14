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

fn CallWithValueExpressionBinding(c: C) {
    Print("Binding scope end");
}

fn Main() -> i32 {
  let c_let: C = {};
  heap.PrintAllocs();

  Print("Bind from c value expression");
  CallWithValueExpressionBinding(c_let);
  heap.PrintAllocs();
  return 0;
}

// CHECK:STDOUT: 0: Heap{}, 1: C{}
// CHECK:STDOUT: Bind from c value expression
// CHECK:STDOUT: Binding scope end
// CHECK:STDOUT: 0: Heap{}, 1: C{}
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: result: 0
