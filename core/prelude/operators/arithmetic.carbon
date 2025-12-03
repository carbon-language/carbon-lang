// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/arithmetic";

import library "prelude/types/int_literal";

// TODO: Per the design, the associated type `Result` in each of these
// interfaces should have a default value of `Self`:
//
//   default let Result:! type = Self;

// TODO: Per the design, for each *With interface there should also be a
// non-With named constraint, such as:
//
//   constraint Add {
//     extend require impls AddWith(Self) where .Result = Self;
//   }

// Addition: `a + b`.
interface AddWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Addition with assignment: `a += b`.
interface AddAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}

// Increment: `++a`.
interface Inc {
  fn Op[ref self: Self]();
}

// Negation: `-a`.
interface Negate {
  let Result:! type;
  fn Op[self: Self]() -> Result;
}

// Subtraction: `a - b`.
interface SubWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Subtraction with assignment: `a -= b`.
interface SubAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}

// Decrement: `--a`.
interface Dec {
  fn Op[ref self: Self]();
}

// Multiplication: `a * b`.
interface MulWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Multiplication with assignment: `a *= b`.
interface MulAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}

// Division: `a / b`.
interface DivWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Division with assignment: `a /= b`.
interface DivAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}

// Modulo: `a % b`.
interface ModWith(Other:! type) {
  let Result:! type;
  fn Op[self: Self](other: Other) -> Result;
}

// Modulo with assignment: `a %= b`.
interface ModAssignWith(Other:! type) {
  fn Op[ref self: Self](other: Other);
}


// Operations for IntLiteral. These need to be here because IntLiteral has no
// associated library of its own.
impl IntLiteral() as AddWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

impl IntLiteral() as DivWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.sdiv";
}

impl IntLiteral() as ModWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.smod";
}

impl IntLiteral() as MulWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.smul";
}

impl IntLiteral() as Negate where .Result = Self {
  fn Op[self: Self]() -> Self = "int.snegate";
}

impl IntLiteral() as SubWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.ssub";
}
