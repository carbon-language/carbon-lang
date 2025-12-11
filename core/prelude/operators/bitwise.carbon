// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/bitwise";

import library "prelude/types/int_literal";

// TODO: Per the design, the associated type `Result` in each of these
// interfaces should have a default value of `Self`:
//
//   default let Result:! type = Self;

// TODO: Per the design, for each *With interface there should also be a
// non-With named constraint, such as:
//
//   constraint BitAnd {
//     extend require impls BitAndWith(Self) where .Result = Self;
//   }

// Bit complement: `^a`.
interface BitComplement {
  let Result:! type;
  fn Op[self: Self]() -> Result;
}

// Bitwise AND: `a & b`.
interface BitAndWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Bitwise AND with assignment: `a &= b`.
interface BitAndAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}

// Bitwise OR: `a | b`.
interface BitOrWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Bitwise OR with assignment: `a |= b`.
interface BitOrAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}

// Bitwise XOR: `a ^ b`.
interface BitXorWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Bitwise XOR with assignment: `a ^= b`.
interface BitXorAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}

// Left shift: `a << b`.
interface LeftShiftWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Left shift with assignment: `a <<= b`.
interface LeftShiftAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}

// Right shift: `a >> b`.
interface RightShiftWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Right shift with assignment: `a >>= b`.
interface RightShiftAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}


// Operations for IntLiteral. These need to be here because IntLiteral has no
// associated library of its own.
impl IntLiteral() as BitAndWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}

impl IntLiteral() as BitComplement where .Result = Self {
  fn Op[self: Self]() -> Self = "int.complement";
}

impl IntLiteral() as BitOrWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}

impl IntLiteral() as BitXorWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}

impl IntLiteral() as LeftShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}

impl IntLiteral() as RightShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}

// Operations for `type`. These need to be here because `type` has no
// associated library of its own.

// Facet type combination.
impl type as BitAndWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "type.and";
}
