// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s

package ExecutableSemanticsTest api;

// This is required to fail even though the Recurse() call's return value isn't
// used.
fn Recurse(x: i32, do_recurse: Bool) -> auto {
  if (do_recurse) {
    // CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/function/auto_return/fail_direct_recurse.carbon:[[@LINE+1]]: Function calls itself, but has a deduced return type
    Recurse(x, false);
  }
  return x;
}

fn Main() -> i32 {
  return Recurse(1, true) - 3;
}
