// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/unformed/static/fail_rhs_def.carbon:[[@LINE+1]]: use of uninitialized variable x
  var y: i32 = x;
  return x;
}
