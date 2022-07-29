// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 1

package ExplorerTest api;

var call_count: i32 = 0;
fn ReturnFalse() -> Bool {
  call_count = call_count + 1;
  return false;
}

fn Main() -> i32 {
  var result: Bool = ReturnFalse() and ReturnFalse();
  return if not result then call_count else -1;
}
