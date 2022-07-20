// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 42

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
