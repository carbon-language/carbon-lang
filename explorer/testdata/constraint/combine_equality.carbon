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

interface I {}
impl i32 as I {}

fn F(A:! i32, B:! i32, C:! i32, D:! i32, E:! i32,
     T:! I where A == B and C == D and C == E and B == D) {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/combine_equality.carbon:[[@LINE+1]]: member access, F not in constraint interface I where .Self:! I is interface I, A:! i32 == B:! i32 == E:! i32 == C:! i32 == D:! i32
  T.F();
}

fn Main() -> i32 {
  F(1, 1, 1, 1, 1, i32);
  return 0;
}
