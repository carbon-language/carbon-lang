// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/uint.carbon

package Core library "prelude/parts/uint";

export import library "prelude/parts/as";
export import library "prelude/parts/int_literal";

private fn MakeUInt(size: IntLiteral()) -> type = "int.make_type_unsigned";

class UInt(N:! IntLiteral()) {
  adapt MakeUInt(N);
}

// Conversions.

impl forall [To:! IntLiteral()] IntLiteral() as ImplicitAs(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert_checked";
}

final impl forall [From:! IntLiteral()] UInt(From) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl forall [To:! IntLiteral()] IntLiteral() as As(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert_checked";
}

final impl forall [From:! IntLiteral()] UInt(From) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// Negate.

final impl forall [N:! IntLiteral()]
    UInt(N) as Negate where .Result = Self {
  fn Op[self: Self]() -> Self = "int.unegate";
}
