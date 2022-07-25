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

var x: i32;

fn Main() -> i32 {
  // CHECK: RUNTIME ERROR: {{.*}}/explorer/testdata/uninitialized/fail_global_uninitialized.carbon:[[@LINE+1]]: undefined behavior: access to uninitialized value Uninit<Placeholder<x>>
  return x;
}
