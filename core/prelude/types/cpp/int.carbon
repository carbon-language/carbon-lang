// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/cpp/int";

import library "prelude/operators";
import library "prelude/types/int";
import library "prelude/types/int_literal";

namespace CppCompat;

// TODO: Add ULong32, LongLong64, ULongLong64.
class CppCompat.Long32 {
  adapt i32;
}

// TODO: Copy.

// Conversions.

// TODO: ImplicitAs from IntLiteral to Long32.
// TODO: ImplicitAs from Long32 to IntLiteral.

// TODO: ImplicitAs from Int(N) to Long32 if N < 32.
impl i32 as ImplicitAs(CppCompat.Long32) {
  fn Convert[self: Self]() -> CppCompat.Long32 = "int.convert_checked";
}

// TODO: ImplicitAs from Long32 to Int(N) if N > 32.
final impl CppCompat.Long32 as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert";
}

// TODO: As from Long32 to Int(N) if N < 32.
// TODO: As from Int(N) to Long32 if N > 32.

// TODO: Operations on Long32.
