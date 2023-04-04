// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package EmptyIdentifier impl;

fn A() {
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/function/fail_recursion_stackoverflow.carbon:[[@LINE+1]]: stack overflow
  A();
}

fn Main() -> i32
{
  A();
  return 0;
}