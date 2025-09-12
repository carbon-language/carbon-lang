// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/copy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/float_literal.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int_literal.carbon

// --- min_prelude/parts/float.carbon

package Core library "prelude/parts/float";

export import library "prelude/parts/as";
export import library "prelude/parts/copy";
export import library "prelude/parts/float_literal";
export import library "prelude/parts/int_literal";

private fn MakeFloat(size: IntLiteral()) -> type = "float.make_type";

class Float(N:! IntLiteral()) {
  adapt MakeFloat(N);
}

impl forall [N:! IntLiteral()] Float(N) as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl forall [To:! IntLiteral()] FloatLiteral() as ImplicitAs(Float(To)) {
  fn Convert[self: Self]() -> Float(To) = "float.convert_checked";
}

// TODO: Remove this once ImplicitAs extends As.
impl forall [To:! IntLiteral()] FloatLiteral() as As(Float(To)) {
  fn Convert[self: Self]() -> Float(To) = "float.convert_checked";
}
