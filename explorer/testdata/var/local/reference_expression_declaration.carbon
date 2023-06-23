// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Initialize c1 from reference expression
// CHECK:STDOUT: 0: Heap{}, 1: C{.id = 1}, 2: C{.id = 1}
// CHECK:STDOUT: End of scope
// CHECK:STDOUT: c1 destroyed
// CHECK:STDOUT: c1 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: !!C{.id = 1}, 2: !!C{.id = 1}
// CHECK:STDOUT:
// CHECK:STDOUT: Initialize c2 from value expression
// CHECK:STDOUT: 0: Heap{}, 1: !!C{.id = 1}, 2: !!C{.id = 1}, 3: C{.id = 2}, 4: C{.id = 2}
// CHECK:STDOUT: End of scope
// CHECK:STDOUT: c2 destroyed
// CHECK:STDOUT: c2 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: !!C{.id = 1}, 2: !!C{.id = 1}, 3: !!C{.id = 2}, 4: !!C{.id = 2}
// CHECK:STDOUT:
// CHECK:STDOUT: Initialize c3 from initializing expression (return <expr>)
// CHECK:STDOUT: c3 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: !!C{.id = 1}, 2: !!C{.id = 1}, 3: !!C{.id = 2}, 4: !!C{.id = 2}, 5: C{.id = 3}, 6: !!3, 7: Uninit<class C>, 8: !!C{.id = 3}
// CHECK:STDOUT: End of scope
// CHECK:STDOUT: c3 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: !!C{.id = 1}, 2: !!C{.id = 1}, 3: !!C{.id = 2}, 4: !!C{.id = 2}, 5: !!C{.id = 3}, 6: !!3, 7: Uninit<class C>, 8: !!C{.id = 3}
// CHECK:STDOUT:
// CHECK:STDOUT: Initialize c4 from initializing expression (returned var)
// CHECK:STDOUT: 0: Heap{}, 1: !!C{.id = 1}, 2: !!C{.id = 1}, 3: !!C{.id = 2}, 4: !!C{.id = 2}, 5: !!C{.id = 3}, 6: !!3, 7: Uninit<class C>, 8: !!C{.id = 3}, 9: C{.id = 4}, 10: !!4
// CHECK:STDOUT: End of scope
// CHECK:STDOUT: c4 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: !!C{.id = 1}, 2: !!C{.id = 1}, 3: !!C{.id = 2}, 4: !!C{.id = 2}, 5: !!C{.id = 3}, 6: !!3, 7: Uninit<class C>, 8: !!C{.id = 3}, 9: !!C{.id = 4}, 10: !!4
// CHECK:STDOUT:
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

fn FromReferenceExpression() {
  var c1_var: C = {.id = 1};
  Print("Initialize c1 from reference expression");
  var c1: C = c1_var;
  heap.PrintAllocs();
  Print("End of scope");
}

fn FromValueExpression() {
  let c2_let: C = {.id = 2};
  Print("Initialize c2 from value expression");
  var c2: C = c2_let;
  heap.PrintAllocs();
  Print("End of scope");
}

fn FromInitializingExpression_ReturnExpr() {
  Print("Initialize c3 from initializing expression (return <expr>)");
  var c3: C = CallWithReturnExpression(3);
  heap.PrintAllocs();
  Print("End of scope");
}

fn FromInitializingExpression_ReturnedVar() {
  Print("Initialize c4 from initializing expression (returned var)");
  var c4: C = CallWithReturnedVar(4);
  heap.PrintAllocs();
  Print("End of scope");
}

fn Main() -> i32 {
  FromReferenceExpression();
  heap.PrintAllocs();
  Print("");

  FromValueExpression();
  heap.PrintAllocs();
  Print("");

  FromInitializingExpression_ReturnExpr();
  heap.PrintAllocs();
  Print("");

  FromInitializingExpression_ReturnedVar();
  heap.PrintAllocs();
  Print("");
  return 0;
}
