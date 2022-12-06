// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32 = 0;
  match (x) {
    // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/match/fail_not_alternative.carbon:[[@LINE+1]]: Alternative pattern must have the form of a field access.
    case i32(n: i32) => {
      return 1;
    }
  }
  return 0;
}
