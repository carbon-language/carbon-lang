// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_self_with_arg.carbon:17: attempt to instantiate an already instantiated generic class: Self(T)

package ExplorerTest api;

class Point(T:! Type) {

  // Error: wrote `Self(T)` instead of `Self`.
  fn Origin(zero: T) -> Self(T) {
    return {.x = zero, .y = zero};
  }

  // Error: wrote `Self(T)` instead of `Self`.
  fn GetX[me: Self(T)]() -> T {
    return me.x;
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  var p: Point(i32) = Point(i32).Origin(0);
  return p.GetX();
}
