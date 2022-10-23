// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn f() {
  __continuation k {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/experimental_continuation/fail_recursive_continuation.carbon:[[@LINE+1]]: 'k' is not usable until after it has been completely declared
  __run k;
  }
}

fn Main() -> i32 {
  return 0;
}
