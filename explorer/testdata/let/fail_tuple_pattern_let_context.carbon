// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  let (var a: auto, b: auto, c: auto, d: auto) = (1, 2, 3, 4);
  a = 0;
  // should fail
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/let/fail_tuple_pattern_let_context.carbon:[[@LINE+1]]: Cannot assign to rvalue 'c'
  c = 0;
  return 0;
}
