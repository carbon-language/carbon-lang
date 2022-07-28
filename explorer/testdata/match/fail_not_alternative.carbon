// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: PROGRAM ERROR: {{.*}}/explorer/testdata/match/fail_not_alternative.carbon:17: Alternative pattern must have the form of a field access.

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32 = 0;
  match (x) {
    case i32(n: i32) => {
      return 1;
    }
  }
  return 0;
}
