// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// Test that a global variable may not depend on a later global.
// Error expected.

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/name_lookup/fail_var_use_before_declaration.carbon:[[@LINE+1]]: 'y' has not been declared yet
var x: i32 = y;

var y: i32 = 0;

fn Main() -> i32 {
  return x;
}
