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

interface Similar(T:! Type) {}

impl forall [T:! Type] T as Similar(T) {}
impl forall [T:! Type] i32 as Similar(T) {}

fn CheckSimilar[T:! Type, U:! Similar(T)](a: T, b: U) {}

fn Main() -> i32 {
  let n: i32 = 0;
  CheckSimilar(true, false);
  CheckSimilar(true, n);
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_param_interface_in_impl.carbon:[[@LINE+1]]: could not find implementation of interface Similar(T = i32) for bool
  CheckSimilar(n, false);
  return 0;
}
