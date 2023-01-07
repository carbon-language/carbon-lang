// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Before New A
// CHECK:STDOUT: CONSTRUCTOR A
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: Before Delete A
// CHECK:STDOUT: DESTRUCTOR A
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A {
  fn Create() -> A {
    Print("CONSTRUCTOR A");
    return {};
  }
  destructor[self: Self] {
    Print("DESTRUCTOR A");
  }
}

fn Main() -> i32 {
  Print("Before New A");
  var pa: A* = heap.New(A.Create());
  Print("Before Delete A");
  heap.Delete(pa);
  return 0;
}
