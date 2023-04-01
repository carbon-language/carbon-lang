// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface HasTypeAndValue {
  let T:! type;
  let V:! T;
}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_rewrite_depends_on_later_rewrite.carbon:[[@LINE+1]]: type error in rewrite constraint: 'i32' is not implicitly convertible to '(X).(HasTypeAndValue.T)'
fn F(X:! HasTypeAndValue where .V = 5 and .T = i32) -> i32 { return X.V; }

impl i32 as HasTypeAndValue where .T = i32 and .V = 5 {}

fn Main() -> i32 {
  return F(i32);
}
