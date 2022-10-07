// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: 921
// CHECK: 922 tuple[1]
// CHECK: (0, true)
// CHECK: 924
// CHECK: 921
// CHECK: 922 tuple[0]
// CHECK: (0, true)
// CHECK: 924
// CHECK: 921
// CHECK: 922 swap((0, true))[1]
// CHECK: (true, 0)
// CHECK: 924
// CHECK: result: 0

package ExplorerTest api;

fn swap[T:! Type, U:! Type](tuple: (T, U)) -> (U, T) {
  return (tuple[1], tuple[0]);
}

fn Main() -> i32 {
  return swap((0, true))[1];
}
