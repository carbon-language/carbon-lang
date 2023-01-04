// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class Point {
  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }
  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_instantiate_non_generic.carbon:[[@LINE+1]]: in call `Point(i32)`, expected callee to be a function, found `type`
  var p: Point(i32) = Point.Origin();
  return 0;
}
