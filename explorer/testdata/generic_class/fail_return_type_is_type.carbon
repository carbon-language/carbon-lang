// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class Point(T:! Type) {
  // The return type should be Point(T). Point by itself is not a type.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_return_type_is_type.carbon:[[@LINE+1]]: 'Point' must be given an argument list
  fn Create(x: T, y: T) -> Point {
    return {.x = x, .y = y};
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  var p: Point(i32) = Point(i32).Create(0, 1);
  return p.x;
}
