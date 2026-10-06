// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int_literal.carbon

// --- min_prelude/parts/array.carbon

package Core library "prelude/parts/array";

export import library "prelude/parts/int_literal";

// Arrays.

private fn MakeArray(element_type: type, bound: IntLiteral) -> type =
    "array.make_type";

class Array(T: type, N: IntLiteral) {
  adapt MakeArray(T, N);
}
