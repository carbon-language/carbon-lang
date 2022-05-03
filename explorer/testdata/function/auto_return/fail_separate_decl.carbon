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

// This declaration is not allowed.
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/auto_return/fail_separate_decl.carbon:[[@LINE+1]]: Function declaration has deduced return type but no body
fn Add(x: i32, y: i32) -> auto;

fn Main() -> i32 {
  return Add(1, 2) - 3;
}
