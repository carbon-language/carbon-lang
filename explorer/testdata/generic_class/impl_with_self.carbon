// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Vector {
  fn Zero() -> Self;
  fn Add[self: Self](b: Self) -> Self;
  fn Scale[self: Self](v: i32) -> Self;
}

class Point(T:! type) {
  var x: T;
  var y: T;
}

external impl Point(i32) as Vector {
  // Allowed: `Self` means `Point(i32)` here.
  fn Zero() -> Self {
    return {.x = 0, .y = 0};
  }

  fn Add[self: Self](b: Self) -> Self {
    return {.x = self.x + b.x, .y = self.y + b.y};
  }

  fn Scale[self: Self](v: i32) -> Self {
    return {.x = self.x * v, .y = self.y * v};
  }
}

fn AddAndScaleGeneric[T:! Vector](a: T, s: i32) -> T {
  return a.Add(T.Zero()).Scale(s);
}

fn Main() -> i32 {
  var a: Point(i32) = {.x = 2, .y = 1};
  var p: Point(i32) = AddAndScaleGeneric(a, 5);
  return p.x - 10;
}
