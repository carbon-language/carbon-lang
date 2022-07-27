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

fn AddInt(a: i32, b: i32) -> auto {
  return a + b;
}

fn Main() -> i32 {
  var x: i32;
  // CHECK: RUNTIME ERROR: {{.*}}/explorer/testdata/uninitialized/fail_local_uninitialized_param.carbon:[[@LINE+1]]: undefined behavior: access to uninitialized value Uninit<Placeholder<x>>
  return AddInt(x, 2);
}
