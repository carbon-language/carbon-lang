// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/bitwise";

import library "prelude/types/int_literal";

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


// Operations for IntLiteral. These need to be here because IntLiteral has no
// associated library of its own.
impl IntLiteral() as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}

impl IntLiteral() as BitComplement {
  fn Op[self: Self]() -> Self = "int.complement";
}

impl IntLiteral() as BitOr {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}

impl IntLiteral() as BitXor {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}

impl IntLiteral() as LeftShift {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}

impl IntLiteral() as RightShift {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}

// Operations for `type`. These need to be here because `type` has no
// associated library of its own.

// Facet type combination.
impl type as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "type.and";
}
