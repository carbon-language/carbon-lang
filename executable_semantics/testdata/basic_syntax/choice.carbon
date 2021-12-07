// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{executable_semantics} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: result: 0

package ExecutableSemanticsTest api;

choice Ints {
  None,
  One(i32),
  Two(i32,i32)
}

fn Main() -> i32 {
  var x: auto = Ints.None();
  var y: auto = Ints.One(42);
  var n: auto = 0;
  match (y) {
    case Ints.None =>
      n = n + 2;
    case Ints.One(x: auto) =>
      n = x + 1 - 42;
    case Ints.Two(a: auto, b: auto) =>
      n = 2;
  }
  match (x) {
    case Ints.One(x: auto) =>
      n = x + 2;
    case Ints.None() =>
      n = n - 1;
    case Ints.Two(x: auto, y: auto) =>
      n = 5;
  }
  return n;
}

// Test some alternate syntaxes
choice MoreInts {
  None(),
  One(i32),
  Two(i32,i32),
}
