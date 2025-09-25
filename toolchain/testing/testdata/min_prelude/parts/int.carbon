// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/copy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int_literal.carbon

// --- min_prelude/parts/int.carbon

package Core library "prelude/parts/int";

export import library "prelude/parts/as";
export import library "prelude/parts/copy";
export import library "prelude/parts/int_literal";

private fn MakeInt(size: IntLiteral()) -> type = "int.make_type_signed";

class Int(N:! IntLiteral()) {
  adapt MakeInt(N);
}

// Copy.

impl forall [N:! IntLiteral()] Int(N) as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

// Conversions.

impl forall [To:! IntLiteral()] IntLiteral() as ImplicitAs(Int(To)) {
  fn Convert[self: Self]() -> Int(To) = "int.convert_checked";
}

final impl forall [From:! IntLiteral()] Int(From) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl forall [To:! IntLiteral()] IntLiteral() as As(Int(To)) {
  fn Convert[self: Self]() -> Int(To) = "int.convert_checked";
}

final impl forall [From:! IntLiteral()] Int(From) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// Negate.

final impl forall [N:! IntLiteral()]
    Int(N) as Negate where .Result = Self {
  fn Op[self: Self]() -> Self = "int.snegate";
}
