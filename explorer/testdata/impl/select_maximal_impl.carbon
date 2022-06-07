// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} --alt_generics %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --alt_generics --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
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

external impl Point(i32) as Number {
  fn Zero() -> Point(i32) { return {.x = 0, .y = 0 }; }
  fn Add[me: Point(i32)](other: Point(i32)) -> Point(i32) {
    return {.x = me.x + other.x, .y = me.y + other.y};
  }
}

// The following two impls are equivalent and would be ambiguous if
// they were the only impls, but the above impl beats them both, so
// there's no problem with this program.
external impl forall [U:! Number] Point(U) as Number {
  fn Zero() -> Point(U) { return {.x = U.Zero(), .y = U.Zero() }; }
  // Broken version of Add. It is not used.
  fn Add[me: Point(U)](other: Point(U)) -> Point(U) {
    return me;
  }
}

external impl forall [U:! Number] Point(U) as Number {
  fn Zero() -> Point(U) { return {.x = U.Zero(), .y = U.Zero() }; }
  // Broken version of Add. It is not used.
  fn Add[me: Point(U)](other: Point(U)) -> Point(U) {
    return me;
  }
}

fn Sum[E:! Number](x: E, y: E) -> E {
  var total: E = E.Zero();
  total = total.Add(x);
  total = total.Add(y);
  return total;
}

fn Main() -> i32 {
  var p: Point(i32) = {.x = 1, .y = 2};
  var q: Point(i32) = {.x = 4, .y = 3};
  var r: Point(i32) = Sum(p, q);
  return r.x - r.y;
}
