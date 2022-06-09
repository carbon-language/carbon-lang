// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn f() {
  __continuation k {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/experimental_continuation/fail_recursive_continuation.carbon:[[@LINE+1]]: 'k' is not usable until after it has been completely declared
  __run k;
  }
}

fn Main() -> i32 {
  return 0;
}
