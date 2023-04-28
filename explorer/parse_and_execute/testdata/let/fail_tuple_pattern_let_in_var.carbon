// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/let/fail_tuple_pattern_let_in_var.carbon:[[@LINE+1]]: syntax error, unexpected LET
  var (var a: auto, b: auto, let c: auto, d: auto) = (1, 2, 3, 4);
  return 0;
}
