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

// This is required to fail even though the Recurse() call's return value isn't
// used.
fn Recurse(x: i32, do_recurse: bool) -> auto {
  if (do_recurse) {
    // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/auto_return/fail_direct_recurse.carbon:[[@LINE+1]]: Function calls itself, but has a deduced return type
    Recurse(x, false);
  }
  return x;
}

fn Main() -> i32 {
  return Recurse(1, true) - 3;
}
