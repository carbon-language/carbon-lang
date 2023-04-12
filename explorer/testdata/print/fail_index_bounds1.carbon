// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest impl;

fn Main() -> i32 {
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/print/fail_index_bounds1.carbon:[[@LINE+1]]: Invalid format string for Print with 1 arg(s) : `Print: {0} {1}`
  Print("Print: {0} {1}", 1);
  return 0;
}
