// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  var t1: (i32, i32) = (5, 2);
  var t2: (i32,) = (5,);
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/tuple/fail_equality_type.carbon:[[@LINE+1]]: (i32, i32) is not equality comparable with (i32) (could not find implementation of interface EqWith(U = (i32)) for (i32, i32))
  if (t1 == t2) {
    return 1;
  } else {
    return 0;
  }
}
