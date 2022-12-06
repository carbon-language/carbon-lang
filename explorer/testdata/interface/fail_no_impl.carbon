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

fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: i32) -> T {
  return a.Add(b).Scale(s);
}

fn Main() -> i32 {
  var a: Point = {.x = 0, .y = 0};
  var b: Point = {.x = 2, .y = 3};
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/interface/fail_no_impl.carbon:[[@LINE+1]]: could not find implementation of interface Vector for class Point
  var p: Point = AddAndScaleGeneric(a, b, 3);
  return p.x - 6;
}
