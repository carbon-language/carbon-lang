// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // error
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/basic_syntax/fail_missing_var.carbon:[[@LINE+1]]: syntax error, unexpected COLON, expecting SLASH or binary *
  x : i32;
  return 1;
}
