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

fn Main() -> i32 {
  returned var x: i32;
  if (0 == 1) {
    x = 0;
  }
  // Static analysis thinks `x` may be formed, defer the check to run-time.
  // CHECK: RUNTIME ERROR: {{.*}}/explorer/testdata/unformed/dynamic/fail_returned_var.carbon:[[@LINE+1]]: undefined behavior: access to uninitialized value Uninit<Placeholder<x>>
  return var;
}
