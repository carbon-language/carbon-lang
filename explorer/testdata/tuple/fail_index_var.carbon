// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  var x: auto = (0, 1);
  var index: i32 = 0;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/tuple/fail_index_var.carbon:[[@LINE+1]]: could not find `index: i32`
  return x[index];
}
