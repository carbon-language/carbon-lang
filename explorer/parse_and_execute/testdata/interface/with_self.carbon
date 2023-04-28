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

class Point {
  var x: i32;
  var y: i32;
  // Allowed: `Self` means `Point` here.
  impl Self as Vector {
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
}

fn AddAndScaleGeneric[T:! Vector](a: T, s: i32) -> T {
  return a.Add(T.Zero()).Scale(s);
}

fn Main() -> i32 {
  var a: Point = {.x = 2, .y = 1};
  var p: Point = AddAndScaleGeneric(a, 5);
  return p.x - 10;
}
