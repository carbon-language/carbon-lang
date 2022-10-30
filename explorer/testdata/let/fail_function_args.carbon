// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn f((var x: i32, y: i32)) -> i32 {
  x = 0;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/let/fail_function_args.carbon:[[@LINE+1]]: Cannot assign to rvalue 'y'
  y = 0;
  return x - 1;
}

fn Main() -> i32 {
  var t: auto = (1, 2);
  return f(t);
}
