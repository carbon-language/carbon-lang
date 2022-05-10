// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/global_variable/fail_init_type_mismatch.carbon:16: type error in initializer of variable: 'Bool' is not implicitly convertible to 'i32'

package ExplorerTest api;

// Test type checking of global variable. Error expected.

var flag: i32 = true;

fn Main() -> i32 {
  return 0;
}
