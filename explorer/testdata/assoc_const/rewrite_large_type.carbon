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
}

interface Splat { fn Op(n: i32) -> Self; }
impl i32 as Splat {
  fn Op(n: i32) -> Self { return n; }
}
impl forall [T:! Splat] (T, T, T) as Splat {
  fn Op(n: i32) -> Self {
    let v: T = T.Op(n);
    return (v, v, v);
  }
}

fn CallSplat(T:! Splat, n: i32) -> T { return T.Op(n); }

fn DoSplat(
    M:! ManyTypes where
      .T0 = (.T1, .T1, .T1) and
      .T1 = (.T2, .T2, .T2) and
      .T2 = (.T3, .T3, .T3) and
      .T3 = (.T4, .T4, .T4) and
      .T4 = (.T5, .T5, .T5) and
      .T5 = (.T6, .T6, .T6) and
      .T6 = (.T7, .T7, .T7) and
      .T7 = i32,
    n: M.T7) -> M.T0 {
  return CallSplat(M.T0, n);
}

interface First { fn Op[me: Self]() -> i32; }
impl i32 as First {
  fn Op[me: Self]() -> i32 { return me; }
}
impl forall [T:! First] (T, T, T) as First {
  fn Op[me: Self]() -> i32 {
    let (a: T, b: T, c: T) = me;
    return a.Op();
  }
}

fn DoFirst(
    M:! ManyTypes where
      .T7 = i32 and
      .T6 = (.T7, .T7, .T7) and
      .T5 = (.T6, .T6, .T6) and
      .T4 = (.T5, .T5, .T5) and
      .T3 = (.T4, .T4, .T4) and
      .T2 = (.T3, .T3, .T3) and
      .T1 = (.T2, .T2, .T2) and
      .T0 = (.T1, .T1, .T1),
    v: M.T0) -> M.T7 {
  return v.(First.Op)();
}

class C {
  impl as ManyTypes where
      .T3 = (.T4, .T4, .T4) and
      .T1 = (.T2, .T2, .T2) and
      .T4 = (.T5, .T5, .T5) and
      .T6 = (.T7, .T7, .T7) and
      .T2 = (.T3, .T3, .T3) and
      .T7 = i32 and
      .T5 = (.T6, .T6, .T6) and
      .T0 = (.T1, .T1, .T1) {}
}

fn Main() -> i32 {
  return DoFirst(C, DoSplat(C, 1));
}
