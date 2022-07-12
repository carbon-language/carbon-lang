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
  fn Add[self: Self](other: Self) -> Self;
}

class Point(T:! Number) {
  fn Origin() -> Point(T) {
    return {.x = T.Zero(), .y = T.Zero()};
  }
  fn Clone[self: Point(T)]() -> Point(T) {
    return {.x = self.x, .y = self.y};
  }
  fn SumXY[self: Point(T)]() -> T {
    return self.x.Add(self.y);
  }
  var x: T;
  var y: T;
}

external impl i32 as Number {
  fn Zero() -> i32 { return 0; }
  fn Add[self: i32](other: i32) -> i32 { return self + other; }
}

fn Main() -> i32 {
  var p: Point(i32) = Point(i32).Origin();
  return p.Clone().SumXY();
}
