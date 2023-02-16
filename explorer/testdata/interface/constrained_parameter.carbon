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
  fn Add[self: Self](o: Self) -> Self;
  fn Mul[self: Self](o: Self) -> Self;
}

external impl i32 as AddMul {
  fn Add[self: i32](o: i32) -> i32 {
    return self + o;
  }
  fn Mul[self: i32](o: i32) -> i32 {
    return self * o;
  }
}

class Holder(T:! AddMul) {
  var v: T;
}

interface Vector(Scalar:! AddMul) {
  fn Zero() -> Self;
  fn Add[self: Self](b: Self) -> Self;
  fn Scale[self: Self](v: Scalar) -> Self;
  fn Hold[self: Self](v: Scalar) -> Holder(Scalar);
}

class Point {
  var x: i32;
  var y: i32;
  impl Point as Vector(i32) {
    fn Zero() -> Point {
      return {.x = 0, .y = 0};
    }
    fn Add[self: Point](b: Point) -> Point {
      return {.x = self.x + b.x, .y = self.y + b.y};
    }
    fn Scale[self: Point](v: i32) -> Point {
      return {.x = self.x * v, .y = self.y * v};
    }
    fn Hold[self: Point](v: i32) -> Holder(i32) {
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
