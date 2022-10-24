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
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: i32) -> Self;
}

class Point {
  var x: i32;
  var y: i32;
  impl Point as Vector {
    fn Zero() -> Point {
      return {.x = 0, .y = 0};
    }
    fn Add[me: Point](b: Point) -> Point {
      return {.x = me.x + b.x, .y = me.y + b.y};
    }
    fn Scale[me: Point](v: i32) -> Point {
      return {.x = me.x * v, .y = me.y * v};
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
