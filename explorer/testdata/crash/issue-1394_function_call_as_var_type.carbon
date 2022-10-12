// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 2>&1 | %{FileCheck} %s
// AUTOUPDATE: %{explorer} %s

package EmptyIdentifier impl;

fn test() -> i32 {
  return 1;
}

fn Main() -> i32 {
  // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/crash/issue-1394_function_call_as_var_type.carbon:[[@LINE+1]]:  function call not allowed here
  var x: test() = 1;
  return 0;
}
