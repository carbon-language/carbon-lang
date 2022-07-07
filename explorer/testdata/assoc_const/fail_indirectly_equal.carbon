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

interface A {
  let T:! Type;
}

fn F1[T:! A where .T == i32](x: T.T) -> i32 {
  // OK: one equality.
  return x;
}

fn F2[U:! A where .T == i32](x: i32) -> U.T {
  // OK: one equality.
  return x;
}

fn F3[T:! A where .T == i32, U:! A where .T == i32](x: T.T) -> U.T {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_indirectly_equal.carbon:[[@LINE+1]]: type error in return value: '(T:! A where .Self.T == i32).T' is not implicitly convertible to '(U:! A where .Self.T == i32).T'
  return x;
}

external impl i32 as A where .T == i32 {}

fn Main() -> i32 {
  return F3(0);
}
