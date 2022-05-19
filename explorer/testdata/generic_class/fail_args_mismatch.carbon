// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false --dump-input=always -v %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

class Point(T:! Type) {
  // Point(T, T) does not match class declaration
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_args_mismatch.carbon:[[@LINE+1]]: wrong number of arguments in function call, expected 1 but got 2
  fn Origin(zero: T) -> Point(T, T) {
    return {.x = zero, .y = zero};
  }

  fn GetX[me: Point(T, T)]() -> T {
    return me.x;
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  var p: Point(i32) = Point(i32).Origin(0);
  return p.GetX();
}
