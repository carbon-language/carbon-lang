// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface AddMul {
  fn Add[me: Self](o: Self) -> Self;
  fn Mul[me: Self](o: Self) -> Self;
}

external impl i32 as AddMul {
  fn Add[me: i32](o: i32) -> i32 {
    return me + o;
  }
  fn Mul[me: i32](o: i32) -> i32 {
    return me * o;
  }
}

class Holder(T:! AddMul) {
  var v: T;
}

interface Vector(Scalar:! AddMul) {
  fn Zero() -> Self;
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: Scalar) -> Self;
  fn Hold[me: Self](v: Scalar) -> Holder(Scalar);
}

class Point {
  var x: i32;
  var y: i32;
  impl Point as Vector(i32) {
    fn Zero() -> Point {
      return {.x = 0, .y = 0};
    }
    fn Add[me: Point](b: Point) -> Point {
      return {.x = me.x + b.x, .y = me.y + b.y};
    }
    fn Scale[me: Point](v: i32) -> Point {
      return {.x = me.x * v, .y = me.y * v};
    }
    fn Hold[me: Point](v: i32) -> Holder(i32) {
      return {.v = v};
    }
  }
}

fn AddAndScaleGeneric[T:! AddMul, U:! Vector(T)](a: U, s: T) -> U {
  return a.Add(U.Zero()).Scale(a.Hold(s).v);
}

fn Main() -> i32 {
  var a: Point = {.x = 2, .y = 1};
  var p: Point = AddAndScaleGeneric(a, 5);
  return p.x - 10;
}
