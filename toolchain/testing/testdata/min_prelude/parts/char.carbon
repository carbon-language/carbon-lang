// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/uint.carbon

// --- min_prelude/parts/char.carbon

package Core library "prelude/parts/char";

import library "prelude/parts/uint";

fn CharLiteral() -> type = "char_literal.make_type";

class Char {
  adapt u8;
}

impl CharLiteral() as ImplicitAs(Char) {
  fn Convert[self: Self]() -> Char = "char.convert_checked";
}
