// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface I {}
impl i32 as I {}

fn F(A:! i32, B:! i32, C:! i32, D:! i32, E:! i32,
     T:! I where A == B and C == D and C == E and B == D) {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_combine_equality.carbon:[[@LINE+1]]: member access, F not in constraint interface I where T is interface I and A == B and C == D and C == E and B == D
  T.F();
}

fn Main() -> i32 {
  F(1, 1, 1, 1, 1, i32);
  return 0;
}
