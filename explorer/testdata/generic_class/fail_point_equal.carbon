// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class Point(T:! type) {
  var x: T;
  var y: T;
}

fn Main() -> i32 {
  var p: Point(i32) = {.x = 0, .y = 0};
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_point_equal.carbon:[[@LINE+1]]: type error in name binding: 'class Point(T = i32)' is not implicitly convertible to 'class Point(T = bool)'
  var q: Point(bool) = p;
  return 0;
}
