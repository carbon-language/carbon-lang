// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

package ExplorerTest api;

// TODO: Implement this with some kind of reflection?
external impl {.x: i32, .y: i32} as EqWith(Self) {
  fn Equal[me: Self](other: Self) -> bool {
    return me.x == other.x and me.y == other.y;
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
