// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 3

package ExplorerTest api;

// Test pausing a continuation with `__await` and restarting it with
// `__run`.

fn Main() -> i32 {
  var x: i32 = 0;
  __continuation k {
    x = x + 1;
    __await;
    x = x + 2;
  }
  __run k;
  __run k;
  return x;
}
