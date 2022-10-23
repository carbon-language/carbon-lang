// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main () -> i32
{
  // 42 cannot be used as the type of a variable.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/basic_syntax/fail_var_type.carbon:[[@LINE+1]]: Expected a type, but got 42
  var x: 42 = 0;
  return x;
}
