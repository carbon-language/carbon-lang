// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/uint";

import library "prelude/types/int_literal";
import library "prelude/operators";

private fn MakeUInt(size: IntLiteral()) -> type = "int.make_type_unsigned";

class UInt(N:! IntLiteral()) {
  adapt MakeUInt(N);
}

impl forall [N:! IntLiteral()] IntLiteral() as ImplicitAs(UInt(N)) {
  fn Convert[self: Self]() -> UInt(N) = "int.convert_checked";
}

impl forall [N:! IntLiteral()] UInt(N) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl forall [N:! IntLiteral()] IntLiteral() as As(UInt(N)) {
  fn Convert[self: Self]() -> UInt(N) = "int.convert_checked";
}

impl forall [N:! IntLiteral()] UInt(N) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Operations.
