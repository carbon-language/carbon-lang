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
// CHECK: 922 t[0]
// CHECK: ((1, 2), (3, 4))
// CHECK: 924
// CHECK: 921
// CHECK: 922 t[0][0]
// CHECK: (1, 2)
// CHECK: 924
// CHECK: 921
// CHECK: 922 t[1]
// CHECK: ((1, 2), (3, 4))
// CHECK: 924
// CHECK: 921
// CHECK: 922 t[1][1]
// CHECK: (3, 4)
// CHECK: 924
// CHECK: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  var t: (auto, (i32, i32)) = ((1, 2), (3, 4));
  return t[0][0] + t[1][1] - 5;
}
