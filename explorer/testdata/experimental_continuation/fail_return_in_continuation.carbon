// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32 = 0;
  __continuation k {
    x = x + 1;
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/experimental_continuation/fail_return_in_continuation.carbon:[[@LINE+1]]: return is not within a function body
    return x;
  }
}
