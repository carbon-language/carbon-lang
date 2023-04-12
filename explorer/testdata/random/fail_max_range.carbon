// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;
fn test(para: i32)->bool{
  return true;
}
fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/random/fail_max_range.carbon:[[@LINE+1]]: Invalid integer literal: 2147483648
  Rand(-2147483648, 2147483647);
  return 0;
}
