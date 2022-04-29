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
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_two_arg_lists.carbon:[[@LINE+1]]: attempt to instantiate an already instantiated generic class: Point(T)(T)
  fn Origin(zero: T) -> Point(T)(T) {
    return {.x = zero, .y = zero};
  }

  fn GetX[me: Point(T)(T)]() -> T {
    return me.x;
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  var p: Point(i32) = Point(i32).Origin(0);
  return p.GetX();
}
