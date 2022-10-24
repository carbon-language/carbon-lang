// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Iface { let N:! i32; }

fn PickType(N: i32) -> Type { return i32; }

fn F[T:! Iface where .N == 5](x: T) -> i32 {
  // TODO: This should be valid: the value of T.N is known to be 5 here.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_unknown_value_specified_in_constraint.carbon:[[@LINE+1]]: value of associated constant (T).(Iface.N) is not known
  var x: PickType(T.N) = 0;
  return x;
}

impl i32 as Iface where .N == 5 {}

fn Main() -> i32 {
  return F(0);
}
