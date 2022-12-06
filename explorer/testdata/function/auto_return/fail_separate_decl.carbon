// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// This declaration is not allowed.
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/function/auto_return/fail_separate_decl.carbon:[[@LINE+1]]: Function declaration has deduced return type but no body
fn ComputeSum(x: i32, y: i32) -> auto;

fn Main() -> i32 {
  return ComputeSum(1, 2) - 3;
}
