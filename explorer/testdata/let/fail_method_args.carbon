// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

class Point {
  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }

  fn SetX[me: Point](x: i32) {
    // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/let/fail_method_args.carbon:[[@LINE+1]]: Cannot assign to rvalue 'x'
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
