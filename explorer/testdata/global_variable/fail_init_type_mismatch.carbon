// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

// Test type checking of global variable. Error expected.

// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/global_variable/fail_init_type_mismatch.carbon:[[@LINE+1]]: type error in initializer of variable: 'bool' is not implicitly convertible to 'i32'
var flag: i32 = true;

fn Main() -> i32 {
  return 0;
}
