// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Iface {
  let T:! type;
}

fn F[T:! Iface where .T == i32](x: T) {}

class Class {
  impl as Iface where .T = i32 {}
}

// OK, constraint on `F` rewritten to `T:! Iface where U == i32`, which we can
// prove from the constraint on `U`.
fn G[U:! type where .Self == i32, T:! Iface where .T = U](x: T, y: U) {
  F(x);
}

// Not OK: would require looking through two levels of `==`.
fn H[V:! type where .Self == i32, W:! Iface where .T == V](x: W, y: V) {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_equal_indirectly.carbon:[[@LINE+1]]: constraint requires that (T).(Iface.T) (with value (W).(Iface.T)) == i32, which is not known to be true
  F(x);
}

fn Main() -> i32 {
  var x: Class = {};
  G(x, 0);
  H(x, 0);
  return 0;
}
