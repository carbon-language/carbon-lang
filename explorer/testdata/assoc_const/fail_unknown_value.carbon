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

interface Iface { let N:! i32; }

fn PickType(N: i32) -> Type { return i32; }

fn F[T:! Iface](x: T) -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_unknown_value.carbon:[[@LINE+1]]: value of associated constant (T:! Iface).N is not known
  var x: PickType(T.N) = 0;
  return x;
}

impl i32 as Iface where .N == 5 {}

fn Main() -> i32 {
  return F(0);
}
