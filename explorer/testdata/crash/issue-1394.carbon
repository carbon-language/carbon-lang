// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 2>&1 | %{FileCheck} %s
// AUTOUPDATE: %{explorer} %s

package EmptyIdentifier impl;

fn apply[T:! Type, U:! Type](f: T, EmptyIdentifier: U)
{
match (true[true]) {}
}

// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/crash/issue-1394.carbon:[[@LINE+1]]:  function call not allowed in this context
fn EmptyIdentifier() -> apply(true, true);

fn Main() -> i32 {
  return 0;
}
