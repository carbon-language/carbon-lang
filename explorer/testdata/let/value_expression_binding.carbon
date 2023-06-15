// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Setup
// CHECK:STDOUT: 0: Heap{}, 1: C{.id = 1}, 2: C{.id = 2}
// CHECK:STDOUT:
// CHECK:STDOUT: Bind from c1 reference expression
// CHECK:STDOUT: c1 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: C{.id = 1}, 2: C{.id = 2}, 3: !!C{.id = 1}
// CHECK:STDOUT:
// CHECK:STDOUT: Bind from c2 value expression
// CHECK:STDOUT: c2 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: C{.id = 1}, 2: C{.id = 2}, 3: !!C{.id = 1}, 4: !!C{.id = 2}
// CHECK:STDOUT:
// CHECK:STDOUT: c2 destroyed
// CHECK:STDOUT: c1 destroyed
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class C {
  fn Create(id: i32) -> C {
    return {.id = id};
  }
  destructor[self: Self] {
    Print("c{0} destroyed", self.id);
  }
  var id: i32;
}

fn CallWithValueExpressionBinding(c: C) {
  return;
}

fn Main() -> i32 {
  Print("Setup");
  var c1_var: C = {.id = 1};
  let c2_let: C = {.id = 2};
  heap.PrintAllocations();
  Print("");

  Print("Bind from c1 reference expression");
  CallWithValueExpressionBinding(c1_var);
  heap.PrintAllocations();
  Print("");

  Print("Bind from c2 value expression");
  CallWithValueExpressionBinding(c2_let);
  heap.PrintAllocations();
  Print("");
  return 0;
}
