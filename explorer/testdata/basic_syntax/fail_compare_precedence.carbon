// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn CompareBools(a: bool, b: bool) -> bool {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/basic_syntax/fail_compare_precedence.carbon:[[@LINE+1]]: syntax error, unexpected EQUAL_EQUAL, expecting SEMICOLON
  return not a == b;
}
