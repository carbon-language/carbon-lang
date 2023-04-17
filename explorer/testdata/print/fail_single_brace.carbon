// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest impl;

fn Main() -> i32 {
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/print/fail_single_brace.carbon:[[@LINE+1]]: `{` must be followed by a second `{` or index in `{`
  Print("{");
  return 0;
}
