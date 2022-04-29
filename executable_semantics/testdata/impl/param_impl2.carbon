// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: result: 0

package ExecutableSemanticsTest api;

interface Number {
  fn Zero() -> Self;
  fn Add[me: Self](other: Self) -> Self;
}

class Point(T:! Number) {
  var x: T;
  var y: T;
}

external impl i32 as Number {
  fn Zero() -> i32 { return 0; }
  fn Add[me: i32](other: i32) -> i32 { return me + other; }
}

external impl forall [U:! Number] Point(U) as Number {
  fn Zero() -> Point(U) { return {.x = U.Zero(), .y = U.Zero() }; }
  fn Add[me: Point(U)](other: Point(U)) -> Point(U) {
    return {.x = me.x.Add(other.x), .y = me.y.Add(other.y)};
  }
}

fn Sum[E:! Number](x: E, y: E) -> E {
  var total: E = E.Zero();
  total = total.Add(x);
  total = total.Add(y);
  return total;
}

fn SumPoints[E:! Number](p: Point(E), q: Point(E)) -> Point(E) {
  return Sum(p, q);
}

fn Main() -> i32 {
  var p: Point(i32) = {.x = 1, .y = 2};
  var q: Point(i32) = {.x = 4, .y = 3};
  var r: Point(i32) = SumPoints(p, q);
  return r.x - r.y;
}
