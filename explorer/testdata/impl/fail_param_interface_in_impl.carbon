// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Similar(T:! type) {}

impl forall [T:! type] T as Similar(T) {}
impl forall [T:! type] i32 as Similar(T) {}

fn CheckSimilar[T:! type, U:! Similar(T)](a: T, b: U) {}

fn Main() -> i32 {
  let n: i32 = 0;
  CheckSimilar(true, false);
  CheckSimilar(true, n);
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_param_interface_in_impl.carbon:[[@LINE+1]]: could not find implementation of interface Similar(T = i32) for bool
  CheckSimilar(n, false);
  return 0;
}
