// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package EmptyIdentifier impl;

fn Main() -> i32 {
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/print/fail_index_text.carbon:[[@LINE+1]]: Invalid format string for Print with 0 arg(s) : `Print: {1a}`
  Print("Print: {1a}");
  return 0;
}
