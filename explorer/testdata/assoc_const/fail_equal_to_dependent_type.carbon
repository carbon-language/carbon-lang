// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Iface {
  let T:! Type;
}

fn F[T:! Iface where .T == i32](x: T) {}

fn G[U:! Iface where .T == i32](x: U) {
  F(x);
}

fn H[V:! Iface](x: V) {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_equal_to_dependent_type.carbon:[[@LINE+1]]: constraint requires that (T).(Iface.T) (with value (V).(Iface.T)) == i32, which is not known to be true
  F(x);
}

class Class {
  impl as Iface where .T = i32 {}
}

fn Main() -> i32 {
  var x: Class = {};
  G(x);
  H(x);
  return 0;
}
