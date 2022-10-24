// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface I {
  fn F() -> i32;
  fn M[me: Self]() -> i32;
}
class A {
  var n: i32;
  impl as I {
    fn F() -> i32 { return 1; }
    fn M[me: Self]() -> i32 { return 2; }
  }
  fn G[me: Self]() -> i32 { return 3; }
}
impl i32 as I {
  fn F() -> i32 { return 4; }
  fn M[me: Self]() -> i32 { return 5; }
}

alias IF = I.F;
alias IM = I.M;
alias AIF = A.(IF);
alias AIM = A.(IM);
alias AG = A.G;
alias i32IF = i32.(IF);
alias i32IM = i32.(IM);

fn Main() -> i32 {
  var a: A = {.n = 0};
  // TODO: Use != once we support it.
  if (not (A.(IF)() == 1)) {
    return 1;
  }
  if (not (a.(IF)() == 1)) {
    return 2;
  }
  if (not (a.(IM)() == 2)) {
    return 3;
  }
  if (not (a.(A.(IM))() == 2)) {
    return 4;
  }
  if (not (AIF() == 1)) {
    return 5;
  }
  if (not (a.(AIM)() == 2)) {
    return 6;
  }
  if (not (a.(AG)() == 3)) {
    return 7;
  }
  if (not (i32.(IF)() == 4)) {
    return 8;
  }
  if (not (0.(IF)() == 4)) {
    return 9;
  }
  if (not (0.(IM)() == 5)) {
    return 10;
  }
  if (not (0.(i32.(IM))() == 5)) {
    return 11;
  }
  if (not (i32IF() == 4)) {
    return 12;
  }
  if (not (0.(i32IM)() == 5)) {
    return 13;
  }
  return 0;
}
