// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A minimal prelude for testing using `Int` or `i32`; required for arrays.
package Core library "prelude";

interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface ImplicitAs(Dest:! type) {
  // TODO: extend As(Dest);
  fn Convert[self: Self]() -> Dest;
}

fn IntLiteral() -> type = "int_literal.make_type";

private fn MakeInt(size: IntLiteral()) -> type = "int.make_type_signed";

class Int(N:! IntLiteral()) {
  adapt MakeInt(N);
}

// Conversions.

impl forall [To:! IntLiteral()] IntLiteral() as ImplicitAs(Int(To)) {
  fn Convert[self: Self]() -> Int(To) = "int.convert_checked";
}

impl forall [From:! IntLiteral()] Int(From) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl forall [To:! IntLiteral()] IntLiteral() as As(Int(To)) {
  fn Convert[self: Self]() -> Int(To) = "int.convert_checked";
}

impl forall [From:! IntLiteral()] Int(From) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// Support testing negative values.

interface Negate {
  fn Op[self: Self]() -> Self;
}

impl IntLiteral() as Negate {
  fn Op[self: Self]() -> Self = "int.snegate";
}
