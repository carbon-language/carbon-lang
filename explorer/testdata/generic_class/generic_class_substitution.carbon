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
  fn Add[me: Self](other: Self) -> Self;
}

class Point(T:! Number) {
  fn Origin() -> Point(T) {
    return {.x = T.Zero(), .y = T.Zero()};
  }
  var x: T;
  var y: T;
}

external impl i32 as Number {
  fn Zero() -> i32 { return 0; }
  fn Add[me: i32](other: i32) -> i32 { return me + other; }
}

fn SumXY[U:! Number](p: Point(U)) -> U {
  return p.Origin().x.Add(p.y);
}

fn Main() -> i32 {
  var p: Point(i32) = {.x = 0, .y = 0};
  return SumXY(p);
}
