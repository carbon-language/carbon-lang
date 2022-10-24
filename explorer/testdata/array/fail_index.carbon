// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  var x: [i32; 2] = (0, 1);
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/array/fail_index.carbon:[[@LINE+1]]: index 2 out of range in (0, 1)
  return x[2];
}
