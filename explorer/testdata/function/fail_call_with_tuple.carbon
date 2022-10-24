// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn f(x: i32, y: i32) -> i32 { return x + y; }

fn Main() -> i32 {
  var xy: (i32, i32) = (1, 2);
  // should fail to type-check
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_call_with_tuple.carbon:[[@LINE+1]]: wrong number of arguments in function call, expected 2 but got 1
  return f(xy);
}
