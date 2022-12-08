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
}

fn ScaleGeneric[T:! Vector](a: T, s: i32) -> T {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/interface/fail_interface_missing_member.carbon:[[@LINE+1]]: member access, Scale not in interface Vector where T is interface Vector
  return a.Scale(s);
}

class Point {
  var x: i32;
  var y: i32;
  impl Point as Vector {
    fn Add[self: Point](b: Point) -> Point {
      return {.x = self.x + b.x, .y = self.y + b.y};
    }
  }
}

fn Main() -> i32 {
  var a: Point = {.x = 3, .y = 1};
  var b: Point = ScaleGeneric(a, 2);
  return b.x - 6;
}
