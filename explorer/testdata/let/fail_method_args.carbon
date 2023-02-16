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

  fn SetX[self: Point](x: i32) {
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/let/fail_method_args.carbon:[[@LINE+1]]: Cannot assign to rvalue 'x'
    x = 10;
  }

  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  var p: Point = Point.Origin();
  p.SetX(42);
  return 0;
}
