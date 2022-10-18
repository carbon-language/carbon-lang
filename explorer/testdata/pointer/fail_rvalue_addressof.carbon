// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32 = 5;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/pointer/fail_rvalue_addressof.carbon:[[@LINE+1]]: Argument to & should be an lvalue.
  var p: i32* = &5;
  return 0;
}
