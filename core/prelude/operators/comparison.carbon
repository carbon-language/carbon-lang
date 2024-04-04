// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/comparison" api;

// Equality comparison: `a == b` and `a != b`.
interface Eq {
  fn Equal[self: Self](other: Self) -> bool;
  fn NotEqual[self: Self](other: Self) -> bool;
}

// Relational comparison: `a < b`, `a <= b`, `a > b`, `a >= b`.
interface Ordered {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool;
  fn LessOrEquivalent[self: Self](other: Self) -> bool;
  fn Greater[self: Self](other: Self) -> bool;
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool;
}
