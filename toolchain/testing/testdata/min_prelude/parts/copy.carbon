// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/copy.carbon

package Core library "prelude/parts/copy";

// Copying an object, which is a conversion from a value representation to an
// initializing representation.
interface Copy {
  fn Op[self: Self]() -> Self;
}

private fn Bool() -> type = "bool.make_type";
private fn CharLiteral() -> type = "char_literal.make_type";
private fn FloatLiteral() -> type = "float_literal.make_type";
private fn IntLiteral() -> type = "int_literal.make_type";

impl Bool() as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl CharLiteral() as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl FloatLiteral() as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl IntLiteral() as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl forall [T:! type] T* as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl type as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl () as Copy {
  fn Op[self: Self]() -> Self = "no_op";
}

impl forall [T:! Copy, U:! Copy] (T, U) as Copy {
  fn Op[self: Self]() -> Self { return (self.0.Op(), self.1.Op()); }
}
