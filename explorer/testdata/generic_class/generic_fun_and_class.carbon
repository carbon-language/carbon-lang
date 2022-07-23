// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

package ExplorerTest api;

interface Number {
  fn Zero() -> Self;
  fn Add[me: Self](other: Self) -> Self;
}

class Point(T:! Number) {
  var x: T;
  var y: T;
}

fn Origin[U :! Number](other: U) -> Point(U) {
  return {.x = U.Zero(), .y = U.Zero()};
}

fn Clone[U :! Number](other: Point(U)) -> Point(U) {
  return {.x = other.x, .y = other.y};
}

fn SumXY[U :! Number](other: Point(U)) -> U {
  return other.x.Add(other.y);
}

external impl i32 as Number {
  fn Zero() -> i32 { return 0; }
  fn Add[me: i32](other: i32) -> i32 { return me + other; }
}

fn Main() -> i32 {
  var p: Point(i32) = Origin(0);
  return SumXY(Clone(p));
}
