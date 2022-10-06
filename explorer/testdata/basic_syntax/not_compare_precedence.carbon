// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn CompareBools(a: bool, b: bool) -> bool {
  // CHECK:SYNTAX ERROR: {{.*}}/explorer/testdata/basic_syntax/not_compare_precedence.carbon:[[@LINE+1]]: syntax error, unexpected EQUAL_EQUAL, expecting SEMICOLON
  return not a == b;
}
