// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// Error: global variable may not be named with keyword `Self`.
// CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/global_variable/fail_named_self.carbon:[[@LINE+1]]: syntax error, unexpected SELF, expecting identifier
var Self: i32 = 0;

fn Main() -> i32 {
  return Self;
}
