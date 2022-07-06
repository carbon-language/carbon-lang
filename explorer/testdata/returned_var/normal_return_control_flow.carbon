// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 3

package ExplorerTest api;

fn AddInt(a: i32, b: i32) -> i32 {
  if (a == b) {
    returned var ret: i32 = a + b;
    return var;
  } else {
    return a + b;
  }
}

fn Main() -> i32 {
  return AddInt(1, 2);
}
