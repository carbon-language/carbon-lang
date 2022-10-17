// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class Shape {
  var x: i32;
}

class Point {

  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }

  fn GetSetX[addr me: Shape*](x: i32) -> i32 {
    var old: auto = (*me).x;
    (*me).x = x;
    return old;
  }

  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  var p: Point = Point.Origin();
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/addr/fail_method_me_type.carbon:[[@LINE+1]]: type error in method access, receiver type: 'class Point' is not implicitly convertible to 'class Shape'
  var x: auto = p.GetSetX(42);
  if (p.x == 42) {
    return x;
  }
  return 1;
}
