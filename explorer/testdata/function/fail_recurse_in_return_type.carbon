// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_recurse_in_return_type.carbon:[[@LINE+1]]: 'T' is not usable until after it has been completely declared
fn T() -> T() { return i32; }

fn Main() -> i32 {
  return 0;
}
