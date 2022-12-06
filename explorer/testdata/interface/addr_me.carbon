// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: 75
// CHECK:STDOUT: 100
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Vector {
  fn Zero() -> Self;
  fn Add[addr me: Self*](b: Self);
  fn Scale[addr me: Self*](v: i32);
}

class Point {
  var x: i32;
  var y: i32;
  impl as Vector {
    fn Zero() -> Self {
      return {.x = 1, .y = 1};
    }
    fn Add[addr me: Self*](b: Self) {
      (*me).x = (*me).x + b.x;
      (*me).y = (*me).y + b.y;
    }
    fn Scale[addr me: Self*](v: i32) {
      (*me).x = (*me).x * v;
      (*me).y = (*me).y * v;
    }
  }
}

fn AddAndScaleGeneric[T:! Vector](p: T*, s: i32) {
  (*p).Add(T.Zero());
  (*p).(Vector.Scale)(s);
  (*p).(T.(Vector.Scale))(s);
}

fn Main() -> i32 {
  var a: Point = {.x = 2, .y = 3};
  AddAndScaleGeneric(&a, 5);
  Print("{0}", a.x);
  Print("{0}", a.y);
  return 0;
}
