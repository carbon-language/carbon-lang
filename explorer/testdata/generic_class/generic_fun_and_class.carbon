// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Number {
  fn Zero() -> Self;
  fn Add[self: Self](other: Self) -> Self;
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
  fn Add[self: i32](other: i32) -> i32 { return self + other; }
}

fn Main() -> i32 {
  var p: Point(i32) = Origin(0);
  return SumXY(Clone(p));
}
