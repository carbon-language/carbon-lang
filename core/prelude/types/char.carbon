// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/char";

import library "prelude/destroy";
import library "prelude/operators";
import library "prelude/types/int_literal";

fn CharLiteral() -> type = "char_literal.make_type";

private fn MakeChar() -> type = "char.make_type";

class Char {
  adapt MakeChar();
}

impl CharLiteral() as ImplicitAs(Char) {
  fn Convert[self: Self]() -> Char = "char.convert_checked";
}

impl Char as ImplicitAs(CharLiteral()) {
  fn Convert[self: Self]() -> CharLiteral() = "char.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl CharLiteral() as As(Char) {
  fn Convert[self: Self]() -> Char = "char.convert_checked";
}

impl Char as As(CharLiteral()) {
  fn Convert[self: Self]() -> CharLiteral() = "char.convert_checked";
}
