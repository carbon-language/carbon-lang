// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/bitwise" api;

// Bit complement: `^a`.
interface BitComplement {
  fn Op[self: Self]() -> Self;
}

// Bitwise AND: `a & b`.
interface BitAnd {
  fn Op[self: Self](other: Self) -> Self;
}

// Bitwise AND with assignment: `a &= b`.
interface BitAndAssign {
  fn Op[addr self: Self*](other: Self);
}

// Bitwise OR: `a | b`.
interface BitOr {
  fn Op[self: Self](other: Self) -> Self;
}

// Bitwise OR with assignment: `a |= b`.
interface BitOrAssign {
  fn Op[addr self: Self*](other: Self);
}

// Bitwise XOR: `a ^ b`.
interface BitXor {
  fn Op[self: Self](other: Self) -> Self;
}

// Bitwise XOR with assignment: `a ^= b`.
interface BitXorAssign {
  fn Op[addr self: Self*](other: Self);
}

// Left shift: `a << b`.
interface LeftShift {
  fn Op[self: Self](other: Self) -> Self;
}

// Left shift with assignment: `a <<= b`.
interface LeftShiftAssign {
  fn Op[addr self: Self*](other: Self);
}

// Right shift: `a >> b`.
interface RightShift {
  fn Op[self: Self](other: Self) -> Self;
}

// Right shift with assignment: `a >>= b`.
interface RightShiftAssign {
  fn Op[addr self: Self*](other: Self);
}
