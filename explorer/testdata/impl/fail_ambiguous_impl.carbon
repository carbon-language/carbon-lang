// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}


package ExplorerTest api;

interface Vector {
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: i32) -> Self;
}

class Point {
  var x: i32;
  var y: i32;
}

external impl Point as Vector {
  fn Add[me: Point](b: Point) -> Point {
      return {.x = me.x + b.x, .y = me.y + b.y};
  }
  fn Scale[me: Point](v: i32) -> Point {
      return {.x = me.x * v, .y = me.y * v};
  }
}

external impl Point as Vector {
  fn Add[me: Point](b: Point) -> Point {
      return {.x = me.x + b.x, .y = me.y + b.y};
  }
  fn Scale[me: Point](v: i32) -> Point {
      return {.x = me.x * v, .y = me.y * v};
  }
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_ambiguous_impl.carbon:[[@LINE+1]]: ambiguous implementations of interface Vector for class Point
}

fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: i32) -> T {
  return a.Add(b).Scale(s);
}

fn Main() -> i32 {
  var a: Point = {.x = 1, .y = 1};
  var b: Point = {.x = 2, .y = 3};
  var p: Point = AddAndScaleGeneric(a, b, 5);
  return p.x - 15;
}
