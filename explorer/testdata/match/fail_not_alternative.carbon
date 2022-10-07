// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32 = 0;
  match (x) {
    // CHECK:SYNTAX ERROR: {{.*}}/explorer/testdata/match/fail_not_alternative.carbon:[[@LINE+1]]: Alternative pattern must have the form of a field access.
    case i32(n: i32) => {
      return 1;
    }
  }
  return 0;
}
