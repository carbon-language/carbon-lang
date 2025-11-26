// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/cpp/int";

import library "prelude/operators";
import library "prelude/types/int";
import library "prelude/types/int_literal";
import library "prelude/types/uint";

namespace CppCompat;

class CppCompat.Long32 {
  adapt i32;
}

class CppCompat.ULong32 {
  adapt u32;
}

class CppCompat.LongLong64 {
  adapt i64;
}

class CppCompat.ULongLong64 {
  adapt u64;
}

// TODO: Copy.

// Conversions.

// TODO: ImplicitAs from IntLiteral to Long32, ULong32, LongLong64, ULongLong64.
// TODO: ImplicitAs from Long32, ULong32, LongLong64, ULongLong64 to IntLiteral.

// TODO: ImplicitAs from Int(N) to Long32 if N < 32.
impl i32 as ImplicitAs(CppCompat.Long32) {
  fn Convert[self: Self]() -> CppCompat.Long32 = "int.convert_checked";
}

// TODO: ImplicitAs from Long32 to Int(N) if N > 32.
final impl CppCompat.Long32 as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert";
}

// TODO: ImplicitAs from UInt(N) to ULong32 if N < 32.
impl u32 as ImplicitAs(CppCompat.ULong32) {
  fn Convert[self: Self]() -> CppCompat.ULong32 = "int.convert_checked";
}

// TODO: ImplicitAs from ULong32 to UInt(N) if N > 32.
final impl CppCompat.ULong32 as ImplicitAs(u32) {
  fn Convert[self: Self]() -> u32 = "int.convert";
}

// TODO: ImplicitAs from Int(N) to LongLong64 if N < 64.
impl i64 as ImplicitAs(CppCompat.LongLong64) {
  fn Convert[self: Self]() -> CppCompat.LongLong64 = "int.convert_checked";
}

// TODO: ImplicitAs from LongLong64 to Int(N) if N > 64.
final impl CppCompat.LongLong64 as ImplicitAs(i64) {
  fn Convert[self: Self]() -> i64 = "int.convert";
}

// TODO: ImplicitAs from UInt(N) to ULongLong64 if N < 64.
impl u64 as ImplicitAs(CppCompat.ULongLong64) {
  fn Convert[self: Self]() -> CppCompat.ULongLong64 = "int.convert_checked";
}

// TODO: ImplicitAs from ULongLong64 to UInt(N) if N > 64.
final impl CppCompat.ULongLong64 as ImplicitAs(u64) {
  fn Convert[self: Self]() -> u64 = "int.convert";
}

// TODO: As from Long32 to Int(N) if N < 32.
// TODO: As from ULong32 to UInt(N) if N < 32.
// TODO: As from LongLong64 to Int(N) if N < 64.
// TODO: As from ULongLong64 to UInt(N) if N < 64.
// TODO: As from Int(N) to Long32 if N > 32.
// TODO: As from UInt(N) to ULong32 if N > 32.
// TODO: As from Int(N) to LongLong64 if N > 64.
// TODO: As from UInt(N) to ULongLong64 if N > 64.

// TODO: Operations on Long32, ULong32, LongLong64, ULongLong64.
