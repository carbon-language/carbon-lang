// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class Point(T:! type) {
  fn Create(x: T, y: T) -> Point(T) {
    return {.x = x, .y = y};
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_generic_class_arg.carbon:[[@LINE+1]]: wrong number of arguments in function call, expected 1 but got 2
  var p: Point(i32) = Point(i32, i32).Create(0, 1);
  return p.x;
}
