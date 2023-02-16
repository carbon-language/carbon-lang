// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class A(T:! type) {}
interface B(T:! type) {}

external impl forall [T:! type] A(T) as B(i32) {}
external impl forall [T:! type] A(i32) as B(T) {}

fn F[T:! B(i32)](x: T) {}
fn G[T:! B(bool)](x: T) {}

fn Main() -> i32 {
  let a: A(bool) = {};
  let b: A(i32) = {};
  F(a);
  G(b);
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_ambiguous_impl_generic.carbon:[[@LINE+1]]: ambiguous implementations of interface B(T = i32) for class A(T = i32)
  F(b);
  return 0;
}
