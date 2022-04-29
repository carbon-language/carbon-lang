// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 2>&1 | %{FileCheck} %s
// AUTOUPDATE: %{executable_semantics} %s

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  var t1: (i32, i32) = (5, 2);
  var t2: (i32,) = (5,);
  // CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/tuple/fail_equality_type.carbon:[[@LINE+1]]: type error in ==
  if (t1 == t2) {
    // CHECK: expected: (i32, i32)
    // CHECK: actual: (i32)
    return 1;
  } else {
    return 0;
  }
}
