// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn AddInt(a: i32, b: i32) -> i32 {
  var ret: i32 = a + b;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/returned_var/fail_missing_declaration.carbon:[[@LINE+1]]: `return var` is not allowed without a returned var defined in scope.
  return var;
}

fn Main() -> i32 {
  return AddInt(1, 2);
}
