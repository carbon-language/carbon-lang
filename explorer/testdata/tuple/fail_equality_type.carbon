// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 2>&1 | %{FileCheck} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn Main() -> i32 {
  var t1: (i32, i32) = (5, 2);
  var t2: (i32,) = (5,);
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/tuple/fail_equality_type.carbon:[[@LINE+1]]: (i32, i32) is not equality comparable with (i32) (mismatch in tuple sizes, expected 2 but got 1)
  if (t1 == t2) {
    return 1;
  } else {
    return 0;
  }
}
