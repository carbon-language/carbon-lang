// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/int_literal.carbon

package Core library "prelude/parts/int_literal";

fn IntLiteral() -> type = "int_literal.make_type";

// Negate.

interface Negate {
  let Result:! type;
  fn Op[self: Self]() -> Result;
}

impl IntLiteral() as Negate where .Result = Self {
  fn Op[self: Self]() -> Self = "int.snegate";
}
