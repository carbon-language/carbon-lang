// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

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
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/interface/fail_impl_missing_member.carbon:[[@LINE+1]]: implementation missing Scale
  }
}

fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: i32) -> T {
  var m: auto = a.Add;
  var n: auto = m(b).Scale;
  return n(s);
}

fn Main() -> i32 {
  var a: Point = {.x = 0, .y = 0};
  var b: Point = {.x = 2, .y = 3};
  var p: Point = AddAndScaleGeneric(a, b, 3);
  return p.x - 6;
}
