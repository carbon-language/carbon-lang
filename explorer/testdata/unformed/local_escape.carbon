// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 42

package ExplorerTest api;

fn AssignIntTo(x: i32, destination: i32*) {
  *destination = x;
}

fn Main() -> i32 {
  var y: i32;
  // `y` is unformed here.
  AssignIntTo(42, &y);
  // `y` is fully formed and usable.
  return y;
}
