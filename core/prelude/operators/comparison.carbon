// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/comparison";

export import library "prelude/types/bool";
import library "prelude/types/int_literal";

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

// Equality comparison for `bool`.
// Note that this must be provided in this library as `bool` doesn't have any
// associated libraries of its own.
impl bool as Eq {
  fn Equal[self: Self](other: Self) -> bool = "bool.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "bool.neq";
}


// Operations for IntLiteral. These need to be here because IntLiteral has no
// associated library of its own.
impl IntLiteral() as Eq {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl IntLiteral() as Ordered {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}
