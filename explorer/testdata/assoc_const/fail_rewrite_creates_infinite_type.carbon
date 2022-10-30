// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface HasType {
  let T:! Type;
}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_rewrite_creates_infinite_type.carbon:[[@LINE+1]]: rewrite of (H).(HasType.T) applies within its own resolved expansion of (H).(HasType.T)*
fn F[H:! HasType where .T = .T*]() {}

fn Main() -> i32 {
  return 0;
}
