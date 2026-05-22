// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/char";

import library "prelude/copy";
import library "prelude/default";
import library "prelude/destroy";
import library "prelude/operators";
import library "prelude/types/uint";
import library "prelude/types/int";
import library "prelude/types/int_literal";

fn CharLiteral() -> type = "char_literal.make_type";

class Char {
  adapt u8;
}

impl Char as UnformedInit {}

impl Char as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl CharLiteral() as ImplicitAs(Char) {
  fn Convert[self: Self]() -> Char = "char.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl CharLiteral() as As(Char) {
  fn Convert[self: Self]() -> Char = "char.convert_checked";
}

impl forall [From:! IntLiteral()] UInt(From) as As(Char) {
  fn Convert[self: Self]() -> Char = "int.convert_char";
}
