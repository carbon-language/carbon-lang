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

class Point(T:! Type) {
  fn Create(x: T, y: T) -> Point(T) {
    return {.x = x, .y = y};
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_generic_class_arg.carbon:[[@LINE+1]]: wrong number of arguments in function call, expected 1 but got 2
  var p: Point(i32) = Point(i32, i32).Create(0, 1);
  return p.x;
}
