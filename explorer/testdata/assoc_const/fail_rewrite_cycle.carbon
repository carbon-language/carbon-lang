// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface ManyTypes {
  let T0:! type;
  let T1:! type;
  let T2:! type;
  let T3:! type;
  let T4:! type;
  let T5:! type;
  let T6:! type;
  let T7:! type;
  let T8:! type;
  let T9:! type;
}

fn F[
  M:! ManyTypes where
    .T0 = .T1 and
    .T1 = .T2 and
    .T2 = .T3 and
    .T3 = .T4 and
    .T4 = .T5 and
    .T5 = .T6 and
    .T6 = .T7 and
    .T7 = .T8 and
    .T8 = .T9 and
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_rewrite_cycle.carbon:[[@LINE+1]]: rewrite of (M).(ManyTypes.T4) applies within its own resolved expansion of (M).(ManyTypes.T4)
    .T9 = .T0]() {}

fn Main() -> i32 {
  return 0;
}
