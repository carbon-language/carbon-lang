// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;
interface Multipliable {
  fn Mul[me: Self](other: Self) -> Self;
}

__mixin Operations for Multipliable {
  export fn Square[me: Self]() -> Self {
    return me.Mul(me);
  }
}

class Point {
  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }

  __mix Operations;

  var x: i32;
  var y: i32;

// CHECK:COMPILATION ERROR: {{.*}}explorer/testdata/mixin/fail_for_constraint.carbon:[[@LINE+1]]: could not find implementation of interface Multipliable for class Point
}

fn Main() -> i32 {
   return 0;
}
