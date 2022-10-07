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
// CHECK: 922 c[0]
// CHECK: (3, 4)
// CHECK: 924
// CHECK: 921
// CHECK: 922 c[1]
// CHECK: (3, 4)
// CHECK: 924
// CHECK: result: 0

package ExplorerTest api;

// Test matching of a tuple inside a tuple.

fn Main() -> i32 {
  var t: auto = ((1, 2), (3, 4));
  match (t) {
    case ((a: auto, b: auto), c: auto) => {
      return a - b + c[0] - c[1] + 2;
    }
  }
  return 1;
}
