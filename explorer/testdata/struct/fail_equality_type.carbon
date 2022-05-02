// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 2>&1 | %{FileCheck} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn Main() -> i32 {
  var t1: {.x: i32, .y: i32} = {.x = 5, .y = 2};
  var t2: {.x: i32,} = {.x = 5,};
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/struct/fail_equality_type.carbon:[[@LINE+3]]: type error in ==
  // CHECK: expected: {.x: i32, .y: i32}
  // CHECK: actual: {.x: i32}
  if (t1 == t2) {
    return 1;
  } else {
    return 0;
  }
}
