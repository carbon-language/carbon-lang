// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class Point(T:! type) {
  fn Origin(zero: T) -> Point(T) {
    return {.x = zero, .y = zero};
  }

  // Error: wrote `Point` instead of `Point(T)`, `Point` by itself is not a type.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_no_args.carbon:[[@LINE+1]]: 'Point' must be given an argument list
  fn GetX[self: Point]() -> T {
    return self.x;
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  var p: Point(i32) = Point(i32).Origin(0);
  return p.GetX();
}
