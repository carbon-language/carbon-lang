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

// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/name_lookup/fail_fn_use_in_param.carbon:[[@LINE+1]]: 'F' is not usable until after it has been completely declared
fn F(x: F(0)) -> Type {
  return i32;
}

fn Main() -> i32 {
  return x;
}
