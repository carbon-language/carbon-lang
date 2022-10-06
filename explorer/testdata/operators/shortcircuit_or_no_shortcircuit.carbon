// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 2

package ExplorerTest api;

var call_count: i32 = 0;
fn ReturnFalse() -> bool {
  call_count = call_count + 1;
  return false;
}

fn Main() -> i32 {
  var result: bool = ReturnFalse() or ReturnFalse();
  return if not result then call_count else -1;
}
