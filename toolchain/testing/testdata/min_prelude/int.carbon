// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon

// --- min_prelude/int.carbon

// A minimal prelude for testing using `Int` or `i32`; required for arrays.
package Core library "prelude";

export import library "prelude/parts/as";

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
