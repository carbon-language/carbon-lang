// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface HasThreeTypes {
  let A:! type;
  let B:! type;
  let C:! type;
  fn Make[self: Self]() -> (A, B, C);
}
fn F[T:! type](x: (T, T, T));
fn G[X:! HasThreeTypes where .A == .B and .B == .C and .C == .A](x: X) {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_multiple_deduction.carbon:[[@LINE+3]]: deduced multiple different values for T:! type:
  // CHECK:STDERR:   (X).(HasThreeTypes.A)
  // CHECK:STDERR:   (X).(HasThreeTypes.B)
  F(x.Make());
}

fn Main() -> i32 { return 0; }
