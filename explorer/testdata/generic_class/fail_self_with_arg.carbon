// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class Point(T:! type) {

  // Error: wrote `Self(T)` instead of `Self`.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_self_with_arg.carbon:[[@LINE+1]]: in call `Self(T)`, expected callee to be a function, found `type`
  fn Origin(zero: T) -> Self(T) {
    return {.x = zero, .y = zero};
  }

  // Error: wrote `Self(T)` instead of `Self`.
  fn GetX[self: Self(T)]() -> T {
    return self.x;
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  var p: Point(i32) = Point(i32).Origin(0);
  return p.GetX();
}
