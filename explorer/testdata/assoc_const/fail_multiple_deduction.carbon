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

interface HasThreeTypes {
  let A:! Type;
  let B:! Type;
  let C:! Type;
  fn Make[me: Self]() -> (A, B, C);
}
fn F[T:! Type](x: (T, T, T));
fn G[X:! HasThreeTypes where .A == .B and .B == .C and .C == .A](x: X) {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_multiple_deduction.carbon:[[@LINE+3]]: deduced multiple different values for T:! Type:
  // CHECK:   (X:! HasThreeTypes where .Self.A == .Self.B and .Self.B == .Self.C and .Self.C == .Self.A).A
  // CHECK:   (X:! HasThreeTypes where .Self.A == .Self.B and .Self.B == .Self.C and .Self.C == .Self.A).B
  F(x.Make());
}

fn Main() -> i32 { return 0; }
