// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/let/fail_tuple_pattern_let_context_nested.carbon:[[@LINE+1]]: syntax error, unexpected LET
  let (var a: auto, b: auto, c: auto, var (d: auto, let e: auto)) =
      (1, 2, 3, (4, 5));
  return 0;
}
