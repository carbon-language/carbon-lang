// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/comparison";

export import library "prelude/types/bool";
import library "prelude/types/int_literal";

// TODO: Per the design, for each *With interface there should also be a
// non-With named constraint, such as:
//
//   constraint Eq {
//     extend require impls EqWith(Self);
//   }

// Equality comparison: `a == b` and `a != b`.
interface EqWith(Other:! type) {
  fn Equal[self: Self](other: Other) -> bool;
  fn NotEqual[self: Self](other: Other) -> bool;
}

// Relational comparison: `a < b`, `a <= b`, `a > b`, `a >= b`.
interface OrderedWith(Other:! type) {
  // TODO: fn Compare
  fn Less[self: Self](other: Other) -> bool;
  fn LessOrEquivalent[self: Self](other: Other) -> bool;
  fn Greater[self: Self](other: Other) -> bool;
  fn GreaterOrEquivalent[self: Self](other: Other) -> bool;
}

// Equality comparison for `bool`.
// Note that this must be provided in this library as `bool` doesn't have any
// associated libraries of its own.
impl bool as EqWith(Self) {
  fn Equal[self: Self](other: Self) -> bool = "bool.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "bool.neq";
}


// Operations for IntLiteral. These need to be here because IntLiteral has no
// associated library of its own.
impl IntLiteral() as EqWith(Self) {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl IntLiteral() as OrderedWith(Self) {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}
