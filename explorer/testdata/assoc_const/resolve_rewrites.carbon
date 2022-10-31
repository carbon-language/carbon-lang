// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1

package ExplorerTest api;

interface ManyTypes {
  let T0:! Type;
  let T1:! Type;
  let T2:! Type;
  let T3:! Type;
  let T4:! Type;
  let T5:! Type;
  let T6:! Type;
  let T7:! Type;
  let T8:! Type;
  let T9:! Type;
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
    .T9 = i32](m: M) -> i32 {
  var v: M.T0 = 1;
  return v;
}

class C {
  impl as ManyTypes where
    .T0 = i32 and
    .T1 = .T0 and
    .T2 = .T1 and
    .T3 = .T2 and
    .T4 = .T3 and
    .T5 = .T4 and
    .T6 = .T5 and
    .T7 = .T6 and
    .T8 = .T7 and
    .T9 = .T8 {}
}

fn Main() -> i32 {
  var c: C = {};
  return F(c);
}
