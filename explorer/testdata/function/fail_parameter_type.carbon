// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// 42 cannot be used as the type of a parameter.
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_parameter_type.carbon:[[@LINE+1]]: type error in type of name binding: 'i32' is not implicitly convertible to 'Type'
fn f(x: 42) -> i32 {
  return x - 1;
}

fn Main() -> i32 {
  return f(1);
}
