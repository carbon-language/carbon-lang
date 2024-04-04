// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/arithmetic" api;

// Addition: `a + b`.
interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

// Addition with assignment: `a += b`.
interface AddAssign {
  fn Op[addr self: Self*](other: Self);
}

// Increment: `++a`.
interface Inc {
  fn Op[addr self: Self*]();
}

// Negation: `-a`.
interface Negate {
  fn Op[self: Self]() -> Self;
}

// Subtraction: `a - b`.
interface Sub {
  fn Op[self: Self](other: Self) -> Self;
}

// Subtraction with assignment: `a -= b`.
interface SubAssign {
  fn Op[addr self: Self*](other: Self);
}

// Decrement: `--a`.
interface Dec {
  fn Op[addr self: Self*]();
}

// Multiplication: `a * b`.
interface Mul {
  fn Op[self: Self](other: Self) -> Self;
}

// Multiplication with assignment: `a *= b`.
interface MulAssign {
  fn Op[addr self: Self*](other: Self);
}

// Division: `a / b`.
interface Div {
  fn Op[self: Self](other: Self) -> Self;
}

// Division with assignment: `a /= b`.
interface DivAssign {
  fn Op[addr self: Self*](other: Self);
}

// Modulo: `a % b`.
interface Mod {
  fn Op[self: Self](other: Self) -> Self;
}

// Modulo with assignment: `a %= b`.
interface ModAssign {
  fn Op[addr self: Self*](other: Self);
}
