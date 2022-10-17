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
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: i32) -> Self;
}

class Point1 {
  var x: i32;
  var y: i32;
  impl Point1 as Vector {
    fn Add[me: Point1](b: Point1) -> Point1 {
      return {.x = me.x + b.x, .y = me.y + b.y};
    }
    fn Scale[me: Point1](v: i32) -> Point1 {
      return {.x = me.x * v, .y = me.y * v};
    }
  }
}

class Point2 {
  var x: i32;
  var y: i32;
  impl Point2 as Vector {
    fn Add[me: Point2](b: Point2) -> Point2 {
      return {.x = me.x + b.x + 1, .y = me.y + b.y + 1};
    }
    fn Scale[me: Point2](v: i32) -> Point2 {
      return {.x = me.x * v * 2, .y = me.y * v * 2};
    }
  }
}

fn ScaleGeneric[U:! Vector](c: U, s: i32) -> U {
  return c.Scale(s);
}

fn AddAndScaleGeneric[T:! Vector, V:! Vector](a: T, b: V, s: i32) -> (T, V) {
  return (ScaleGeneric(a.Add(a), s),
  	  ScaleGeneric(b.Add(b), s));
}

fn Main() -> i32 {
  var a: Point1 = {.x = 1, .y = 1};
  var b: Point2 = {.x = 2, .y = 3};
  var (p: Point1, q: Point2) = AddAndScaleGeneric(a, b, 5);
  return q.x - p.x - 40;
}
