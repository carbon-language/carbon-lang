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
  fn Add[self: Self](b: Self) -> Self;
  fn Scale[self: Self](v: i32) -> Self;
}

class Point {
  var x: i32;
  var y: i32;
  impl Point as Vector {
    fn Add[self: Point](b: Point) -> Point {
      return {.x = self.x + b.x, .y = self.y + b.y};
    }
    fn Scale[self: Point](v: i32) -> Point {
      return {.x = self.x * v, .y = self.y * v};
    }
  }
}

fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: i32) -> T {
  var m: __Fn(T)->T = a.Add;
  var n: __Fn(i32)->T = m(b).Scale;
  return n(s);
}

fn Main() -> i32 {
  var a: Point = {.x = 1, .y = 1};
  var b: Point = {.x = 2, .y = 3};
  var p: Point = AddAndScaleGeneric(a, b, 5);
  return p.x - 15;
}
