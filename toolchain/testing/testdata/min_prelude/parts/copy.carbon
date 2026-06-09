// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/copy.carbon

package Core library "prelude/parts/copy";

// Copying an object, which is a conversion from a value representation to an
// initializing representation.
interface Copy {
  fn Op(self) -> Self;
}

private fn MakeBool() -> type = "bool.make_type";
private alias Bool = MakeBool();
private fn MakeCharLiteral() -> type = "char_literal.make_type";
private alias CharLiteral = MakeCharLiteral();
private fn MakeFloatLiteral() -> type = "float_literal.make_type";
private alias FloatLiteral = MakeFloatLiteral();
private fn MakeIntLiteral() -> type = "int_literal.make_type";
private alias IntLiteral = MakeIntLiteral();

impl forall [T:! Copy] const T as Copy {
  fn Op(self) -> Self { return (self as T).(Copy.Op)() as const T; }
}

impl Bool as Copy {
  fn Op(self) -> Self = "primitive_copy";
}

impl CharLiteral as Copy {
  fn Op(self) -> Self = "primitive_copy";
}

impl FloatLiteral as Copy {
  fn Op(self) -> Self = "primitive_copy";
}

impl IntLiteral as Copy {
  fn Op(self) -> Self = "primitive_copy";
}

impl forall [T:! type] T* as Copy {
  fn Op(self) -> Self = "primitive_copy";
}

impl type as Copy {
  fn Op(self) -> Self = "primitive_copy";
}

impl () as Copy {
  fn Op(self) -> Self = "no_op";
}

impl forall [T:! Copy, U:! Copy] (T, U) as Copy {
  fn Op(self) -> Self {
    return (self.0.Op(), self.1.Op());
  }
}

impl forall [T:! Copy, U:! Copy, V:! Copy] (T, U, V) as Copy {
  fn Op(self) -> Self {
    return (self.0.Op(), self.1.Op(), self.2.Op());
  }
}
