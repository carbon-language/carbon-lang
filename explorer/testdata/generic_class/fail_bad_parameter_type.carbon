// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class Point(T:! i32) {
  fn Origin(zero: T) -> Point(T) {
    return {.x = zero, .y = zero};
  }

  fn GetX[me: Point(T)]() -> T {
    return me.x;
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_bad_parameter_type.carbon:[[@LINE+1]]: type error in call: 'Type' is not implicitly convertible to 'i32'
  var p: Point(i32) = Point(i32).Origin(0);
  return p.GetX();
}
