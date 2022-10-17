// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1

package ExplorerTest api;

var call_count: i32 = 0;
fn ReturnTrue() -> bool {
  call_count = call_count + 1;
  return true;
}

fn Main() -> i32 {
  var result: bool = ReturnTrue() or ReturnTrue();
  return if result then call_count else -1;
}
