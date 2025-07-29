// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/int";

import library "prelude/destroy";
import library "prelude/operators";
import library "prelude/types/int_literal";

private fn MakeInt(size: IntLiteral()) -> type = "int.make_type_signed";

class Int(N:! IntLiteral()) {
  adapt MakeInt(N);
}

// Conversions.

impl forall [To:! IntLiteral()] IntLiteral() as ImplicitAs(Int(To)) {
  fn Convert[self: Self]() -> Int(To) = "int.convert_checked";
}

final impl forall [From:! IntLiteral()] Int(From) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl forall [To:! IntLiteral()] IntLiteral() as As(Int(To)) {
  fn Convert[self: Self]() -> Int(To) = "int.convert_checked";
}

final impl forall [From:! IntLiteral()] Int(From) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Allow as an implicit conversion if N > M.
final impl forall [From:! IntLiteral(), To:! IntLiteral()] Int(From) as As(Int(To)) {
  fn Convert[self: Self]() -> Int(To) = "int.convert";
}

// Comparisons.

// - Int(N) vs Int(M).

final impl forall [N:! IntLiteral(), M:! IntLiteral()] Int(N) as EqWith(Int(M)) {
  fn Equal[self: Self](other: Int(M)) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Int(M)) -> bool = "int.neq";
}

final impl forall [N:! IntLiteral(), M:! IntLiteral()] Int(N) as OrderedWith(Int(M)) {
  // TODO: fn Compare
  fn Less[self: Self](other: Int(M)) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Int(M)) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Int(M)) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Int(M)) -> bool = "int.greater_eq";
}

// - Int(N) on left-hand side.

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))] Int(N) as EqWith(T) {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))] Int(N) as OrderedWith(T) {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

// - Int(N) on right-hand side.

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))] T as EqWith(Int(N)) {
  fn Equal[self: Int(N)](other: Int(N)) -> bool = "int.eq";
  fn NotEqual[self: Int(N)](other: Int(N)) -> bool = "int.neq";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))] T as OrderedWith(Int(N)) {
  // TODO: fn Compare
  fn Less[self: Int(N)](other: Int(N)) -> bool = "int.less";
  fn LessOrEquivalent[self: Int(N)](other: Int(N)) -> bool = "int.less_eq";
  fn Greater[self: Int(N)](other: Int(N)) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Int(N)](other: Int(N)) -> bool = "int.greater_eq";
}

// Arithmetic.

// - Homogeneous.

final impl forall [N:! IntLiteral()]
    Int(N) as AddWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

final impl forall [N:! IntLiteral()]
    Int(N) as DivWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.sdiv";
}

final impl forall [N:! IntLiteral()]
    Int(N) as ModWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.smod";
}

final impl forall [N:! IntLiteral()]
    Int(N) as MulWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.smul";
}

final impl forall [N:! IntLiteral()]
    Int(N) as Negate where .Result = Self {
  fn Op[self: Self]() -> Self = "int.snegate";
}

final impl forall [N:! IntLiteral()]
    Int(N) as SubWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.ssub";
}

// - Int(N) on left-hand side.

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as AddWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as DivWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as ModWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as MulWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as SubWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
}

// - Int(N) on right-hand side.

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as AddWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as DivWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as ModWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as MulWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as SubWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
}

// Bitwise operators.

// - Homogeneous.

final impl forall [N:! IntLiteral()]
    Int(N) as BitAndWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}

final impl forall [N:! IntLiteral()]
    Int(N) as BitComplement where .Result = Self {
  fn Op[self: Self]() -> Self = "int.complement";
}

final impl forall [N:! IntLiteral()]
    Int(N) as BitOrWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}

final impl forall [N:! IntLiteral()]
    Int(N) as BitXorWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}

final impl forall [N:! IntLiteral()]
    Int(N) as LeftShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}

final impl forall [N:! IntLiteral()]
    Int(N) as RightShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}

// - Int(N) on left-hand side.

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as BitAndWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as BitOrWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as BitXorWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as LeftShiftWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as RightShiftWith(U) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
}

// - Int(N) on right-hand side.

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as BitAndWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as BitOrWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as BitXorWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as LeftShiftWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(Int(N))]
    T as RightShiftWith(Int(N)) where .Result = Int(N) {
  fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
}

// Compound assignments.

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as AddAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.sadd_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as BitAndAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.and_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as BitOrAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.or_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as BitXorAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.xor_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as DivAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.sdiv_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as LeftShiftAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.left_shift_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as ModAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.smod_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as MulAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.smul_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as RightShiftAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.right_shift_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(Int(N))]
    Int(N) as SubAssignWith(U) {
  fn Op[addr self: Self*](other: Self) = "int.ssub_assign";
}

// Increment and decrement.

// TODO: Add builtins for these to avoid emitting a call.

impl forall [N:! IntLiteral()] Int(N) as Dec {
  fn Op[addr self: Self*]() {
    // TODO: `*self -= 1;` should work, but fails because the `impl IntLiteral
    // as ImplicitAs(Int(N))` is not final, which causes us to not attempt the
    // conversion from `1` to `Int(N)` until runtime, when we've lost the
    // constant value.
    fn AsInt(n: IntLiteral()) -> Int(N) = "int.convert_checked";
    *self -= AsInt(1);
  }
}

impl forall [N:! IntLiteral()] Int(N) as Inc {
  fn Op[addr self: Self*]() {
    fn AsInt(n: IntLiteral()) -> Int(N) = "int.convert_checked";
    *self += AsInt(1);
  }
}
