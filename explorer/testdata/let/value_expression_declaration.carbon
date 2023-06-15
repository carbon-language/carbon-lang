// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Setup
// CHECK:STDOUT: 0: Heap{}, 1: C{.id = 1}, 2: C{.id = 2}
// CHECK:STDOUT:
// CHECK:STDOUT: Initialize c1 from reference expression
// CHECK:STDOUT: 0: Heap{}, 1: C{.id = 1}, 2: C{.id = 2}
// CHECK:STDOUT:
// CHECK:STDOUT: Initialize c2 from value expression
// CHECK:STDOUT: 0: Heap{}, 1: C{.id = 1}, 2: C{.id = 2}, 3: C{.id = 2}
// CHECK:STDOUT:
// CHECK:STDOUT: Initialize c3 from initializing expression (return <expr>)
// CHECK:STDOUT: c3 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: C{.id = 1}, 2: C{.id = 2}, 3: C{.id = 2}, 4: C{.id = 3}, 5: !!3, 6: !!C{.id = 3}
// CHECK:STDOUT:
// CHECK:STDOUT: Initialize c4 from initializing expression (returned var)
// CHECK:STDOUT: 0: Heap{}, 1: C{.id = 1}, 2: C{.id = 2}, 3: C{.id = 2}, 4: C{.id = 3}, 5: !!3, 6: !!C{.id = 3}, 7: C{.id = 4}, 8: !!4
// CHECK:STDOUT:
// CHECK:STDOUT: c4 destroyed
// CHECK:STDOUT: c3 destroyed
// CHECK:STDOUT: c2 destroyed
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

fn CallWithReturnExpression(id: i32) -> C {
  var c: C = {.id = id};
  return c;
}

fn CallWithReturnedVar(id: i32) -> C {
  returned var c: C = {.id = id};
  return var;
}

fn Main() -> i32 {
  Print("Setup");
  var c1_var: C = {.id = 1};
  let c2_let: C = {.id = 2};
  heap.PrintAllocations();
  Print("");

  Print("Initialize c1 from reference expression");
  let c1: C = c1_var;
  heap.PrintAllocations();
  Print("");

  Print("Initialize c2 from value expression");
  let c2: C = c2_let;
  heap.PrintAllocations();
  Print("");

  Print("Initialize c3 from initializing expression (return <expr>)");
  let c3: C = CallWithReturnExpression(3);
  heap.PrintAllocations();
  Print("");

  Print("Initialize c4 from initializing expression (returned var)");
  let c4: C = CallWithReturnedVar(4);
  heap.PrintAllocations();
  Print("");
  return 0;
}
