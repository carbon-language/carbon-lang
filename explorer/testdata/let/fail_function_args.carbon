// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn f((var x: i32, y: i32)) -> i32 {
  x = 0;
  // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/let/fail_function_args.carbon:[[@LINE+1]]: Cannot assign to rvalue 'y'
  y = 0;
  return x - 1;
}

fn Main() -> i32 {
  var t: auto = (1, 2);
  return f(t);
}
