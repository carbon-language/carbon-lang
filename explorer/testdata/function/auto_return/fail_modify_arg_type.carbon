// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn Id(t: Type) -> auto { return t; }

// Test non-trivial type expression in parameter type.

// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/function/auto_return/fail_modify_arg_type.carbon:[[@LINE+1]]:  function call not allowed here
fn f(x: Id(i32)) -> i32 {
  return x;
}

fn Main() -> i32 {
  return f(0);
}
