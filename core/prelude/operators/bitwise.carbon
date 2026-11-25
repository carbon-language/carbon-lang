// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/bitwise";

import library "prelude/types/int_literal";

// TODO: Per the design, the associated type `Result` in each of these
// interfaces should have a default value of `Self`:
//
//   default let Result: type = Self;

// Bit complement: `^a`.
interface BitComplement {
  let Result: type;
  fn Op(self) -> Result;
}

// Bitwise AND: `a & b`.
interface BitAndWith(Other: type) {
  let Result: type;
  fn Op(self, other: Other) -> Result;
}

constraint BitAnd {
  extend require impls BitAndWith(Self) where .Result = Self;
}

// Bitwise AND with assignment: `a &= b`.
interface BitAndAssignWith(Other: type) {
  fn Op(ref self, other: Other);
}

constraint BitAndAssign {
  extend require impls BitAndAssignWith(Self);
}

// Bitwise OR: `a | b`.
interface BitOrWith(Other: type) {
  let Result: type;
  fn Op(self, other: Other) -> Result;
}

constraint BitOr {
  extend require impls BitOrWith(Self) where .Result = Self;
}

// Bitwise OR with assignment: `a |= b`.
interface BitOrAssignWith(Other: type) {
  fn Op(ref self, other: Other);
}

constraint BitOrAssign {
  extend require impls BitOrAssignWith(Self);
}

// Bitwise XOR: `a ^ b`.
interface BitXorWith(Other: type) {
  let Result: type;
  fn Op(self, other: Other) -> Result;
}

constraint BitXor {
  extend require impls BitXorWith(Self) where .Result = Self;
}

// Bitwise XOR with assignment: `a ^= b`.
interface BitXorAssignWith(Other: type) {
  fn Op(ref self, other: Other);
}

constraint BitXorAssign {
  extend require impls BitXorAssignWith(Self);
}

// Left shift: `a << b`.
interface LeftShiftWith(Other: type) {
  let Result: type;
  fn Op(self, other: Other) -> Result;
}

constraint LeftShift {
  extend require impls LeftShiftWith(Self) where .Result = Self;
}

// Left shift with assignment: `a <<= b`.
interface LeftShiftAssignWith(Other: type) {
  fn Op(ref self, other: Other);
}

constraint LeftShiftAssign {
  extend require impls LeftShiftAssignWith(Self);
}

// Right shift: `a >> b`.
interface RightShiftWith(Other: type) {
  let Result: type;
  fn Op(self, other: Other) -> Result;
}

constraint RightShift {
  extend require impls RightShiftWith(Self) where .Result = Self;
}

// Right shift with assignment: `a >>= b`.
interface RightShiftAssignWith(Other: type) {
  fn Op(ref self, other: Other);
}

constraint RightShiftAssign {
  extend require impls RightShiftAssignWith(Self);
}


// Operations for IntLiteral. These need to be here because IntLiteral has no
// associated library of its own.
impl IntLiteral as BitAndWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "int.and";
}

impl IntLiteral as BitComplement where .Result = Self {
  fn Op(self) -> Self = "int.complement";
}

impl IntLiteral as BitOrWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "int.or";
}

impl IntLiteral as BitXorWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "int.xor";
}

impl IntLiteral as LeftShiftWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "int.left_shift";
}

impl IntLiteral as RightShiftWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "int.right_shift";
}

// Operations for `type`. These need to be here because `type` has no
// associated library of its own.

// Facet type combination.
impl type as BitAndWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "type.and";
}
