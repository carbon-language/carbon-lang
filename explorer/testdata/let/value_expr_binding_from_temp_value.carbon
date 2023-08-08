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
  heap.PrintAllocs();

  Print("Bind from 'temporary' c value expression");
  CallWithValueExpressionBinding({});
  heap.PrintAllocs();
  return 0;
}

// CHECK:STDOUT: 0: Heap{}
// CHECK:STDOUT: Bind from 'temporary' c value expression
// CHECK:STDOUT: Binding scope end
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: 0: Heap{}, 1: !!C{}
// CHECK:STDOUT: result: 0
