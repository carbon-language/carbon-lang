// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

// TODO: Implement this with some kind of reflection?
external impl {.x: i32, .y: i32} as EqWith(Self) {
  fn Equal[me: Self](other: Self) -> bool {
    return me.x == other.x and me.y == other.y;
  }
  fn NotEqual[me: Self](other: Self) -> bool {
    return me.x != other.x or me.y != other.y;
  }
}

fn Main() -> i32 {
  var t1: {.x: i32, .y: i32} = {.x = 5, .y = 2};
  var t2: {.x: i32, .y: i32} = {.x = 5, .y = 4};
  if (t1 == t2) {
    return 1;
  } else {
    return 0;
  }
}
