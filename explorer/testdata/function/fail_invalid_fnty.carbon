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

// TODO: We should type-check the prelude before the main input so that we can
// use it within the types of top-level declarations.
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_invalid_fnty.carbon:[[@LINE+2]]: type error in `-`:
// CHECK: missing declaration for builtin `Negate`
fn f(g: __Fn(-true) -> true) {
}

fn Main() -> i32 {
  return 0;
}
