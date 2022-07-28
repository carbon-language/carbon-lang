// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 2>&1 | %{FileCheck} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/comparison/fail_no_impl.carbon:[[@LINE+1]]: i32 is not equality comparable with String (could not find implementation of interface EqWith(U = String) for i32)
  Print("different types equal: {0}", if 1 == "1" then 1 else 0);
  return 0;
}
