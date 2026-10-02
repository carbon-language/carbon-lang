// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/array";

import library "prelude/default";
import library "prelude/types/int_literal";

private fn MakeArray(T: type, N: IntLiteral) -> type = "array.make_type";

// The type `array(T, N)`.
//
// Note that `array(T, N)` syntax can't be used within this file, because it
// refers to this class.
class Array(T: type, N: IntLiteral) {
  adapt MakeArray(T, N);
}

impl forall [T: UnformedInit, N: IntLiteral] Array(T, N) as UnformedInit {}
