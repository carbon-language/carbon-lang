// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package EmptyIdentifier impl;

fn test() -> i32 {
  return 1;
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_var_type_is_call.carbon:[[@LINE+1]]: Expected a type, but got 1
  var x: test() = 1;
  return 0;
}
