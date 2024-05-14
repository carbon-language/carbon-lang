// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  fn Create() -> C {
    heap.PrintAllocs();
    return Create2();
  }
  fn Create2() -> C {
    return {};
  }
  destructor[self: Self] {
    Print("c destroyed");
  }
}

fn CallWithReferenceExpressionBinding(var c: C) {
    heap.PrintAllocs();
    Print("Binding scope end");
}

fn Main() -> i32 {
  Print("Bind from c initializing expression");
  CallWithReferenceExpressionBinding(C.Create());
  heap.PrintAllocs();
  return 0;
}

// CHECK:STDOUT: Bind from c initializing expression
// CHECK:STDOUT: 0: Heap{}, 1: !Uninit<class C>
// CHECK:STDOUT: 0: Heap{}, 1: C{}, 2: C{}
// CHECK:STDOUT: Binding scope end
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: c destroyed
// CHECK:STDOUT: 0: Heap{}, 1: !!C{}, 2: !!C{}
// CHECK:STDOUT: result: 0
