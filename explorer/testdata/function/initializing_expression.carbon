// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 0: Heap{}
// CHECK:STDOUT: No returned value
// CHECK:STDOUT: 0: Heap{}
// CHECK:STDOUT: Reference expressions from Initializing expression
// CHECK:STDOUT: a1 destroyed
// CHECK:STDOUT: a2 destroyed
// CHECK:STDOUT: a1 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: !!1, 2: !!1, 3: !!A{.index = 1}, 4: !!A{.index = 1}, 5: !!3, 6: !!A{.index = 2}
// CHECK:STDOUT: Value expressions from Initializing expression
// CHECK:STDOUT: a1 destroyed
// CHECK:STDOUT: a2 destroyed
// CHECK:STDOUT: a1 destroyed
// CHECK:STDOUT: 0: Heap{}, 1: !!1, 2: !!1, 3: !!A{.index = 1}, 4: !!A{.index = 1}, 5: !!3, 6: !!A{.index = 2}, 7: !!1, 8: !!1, 9: !!A{.index = 1}, 10: !!A{.index = 1}, 11: !!3, 12: !!A{.index = 2}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A {
  destructor[self: Self] {
    Print("a{0} destroyed", self.index);
  }
  var index: i32;
}

fn CallWithNoReturn() {
  return;
}

fn CallWithReturnExpression() -> i32 {
  var c: i32 = 1;
  return c;
}

fn CallWithReturnExpressionClass() -> A {
  var a: A = {.index = 0};
  a.index = 1;
  return a;
}

fn CallWithReturnedVar() -> i32 {
  returned var c: i32 = 0;
  c = 3;
  return var;
}

fn CallWithReturnedVarClass() -> A {
  returned var a: A = {.index = 1};
  a.index = 2;
  return var;
}

fn TestNoReturn() {
  Print("No returned value");
  CallWithNoReturn();
}

fn TestInitializeReferenceExpr() {
  // Reference expressions from Initializing expression
  Print("Reference expressions from Initializing expression");
  var vn1: i32 = CallWithReturnExpression();
  var va1: A = CallWithReturnExpressionClass();
  var vn2: i32 = CallWithReturnedVar();
  var va2: A = CallWithReturnedVarClass();
}

fn TestInitializeValueExpr() {
  // Value expressions from Initializing expression
  Print("Value expressions from Initializing expression");
  let ln1: i32 = CallWithReturnExpression();
  let la1: A = CallWithReturnExpressionClass();
  let ln2: i32 = CallWithReturnedVar();
  let la2: A = CallWithReturnedVarClass();
}

fn Main() -> i32 {
  heap.PrintAllocations();

  TestNoReturn();
  heap.PrintAllocations();

  TestInitializeReferenceExpr();
  heap.PrintAllocations();

  TestInitializeValueExpr();
  heap.PrintAllocations();
  return 0;
}
