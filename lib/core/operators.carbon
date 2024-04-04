// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "operators" api;

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}
interface AddAssign {
  fn Op[addr self: Self*](other: Self);
}

interface BitAnd {
  fn Op[self: Self](other: Self) -> Self;
}
interface BitAndAssign {
  fn Op[addr self: Self*](other: Self);
}

interface BitComplement {
  fn Op[self: Self]() -> Self;
}

interface BitOr {
  fn Op[self: Self](other: Self) -> Self;
}
interface BitOrAssign {
  fn Op[addr self: Self*](other: Self);
}

interface BitXor {
  fn Op[self: Self](other: Self) -> Self;
}
interface BitXorAssign {
  fn Op[addr self: Self*](other: Self);
}

interface Dec {
  fn Op[addr self: Self*]();
}

interface Div {
  fn Op[self: Self](other: Self) -> Self;
}
interface DivAssign {
  fn Op[addr self: Self*](other: Self);
}

interface Eq {
  fn Equal[self: Self](other: Self) -> bool;
  fn NotEqual[self: Self](other: Self) -> bool;
}

interface Inc {
  fn Op[addr self: Self*]();
}

interface LeftShift {
  fn Op[self: Self](other: Self) -> Self;
}
interface LeftShiftAssign {
  fn Op[addr self: Self*](other: Self);
}

interface Mod {
  fn Op[self: Self](other: Self) -> Self;
}
interface ModAssign {
  fn Op[addr self: Self*](other: Self);
}

interface Mul {
  fn Op[self: Self](other: Self) -> Self;
}
interface MulAssign {
  fn Op[addr self: Self*](other: Self);
}

interface Negate {
  fn Op[self: Self]() -> Self;
}

interface Ordered {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool;
  fn LessOrEquivalent[self: Self](other: Self) -> bool;
  fn Greater[self: Self](other: Self) -> bool;
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool;
}

interface RightShift {
  fn Op[self: Self](other: Self) -> Self;
}
interface RightShiftAssign {
  fn Op[addr self: Self*](other: Self);
}

interface Sub {
  fn Op[self: Self](other: Self) -> Self;
}
interface SubAssign {
  fn Op[addr self: Self*](other: Self);
}
