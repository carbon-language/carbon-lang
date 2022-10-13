// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn Main() -> i32 {
  return 0;
}

// CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/import/fail_order.carbon:[[@LINE+1]]: syntax error, unexpected IMPORT, expecting END_OF_FILE
import ExplorerTest library "Nonexistent";
