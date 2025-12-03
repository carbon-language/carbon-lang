// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/uint";

import library "prelude/copy";
import library "prelude/destroy";
import library "prelude/operators";
import library "prelude/types/int";
import library "prelude/types/int_literal";

private fn MakeUInt(size: IntLiteral()) -> type = "int.make_type_unsigned";

class UInt(N:! IntLiteral()) {
  adapt MakeUInt(N);
}

// Copy.

impl forall [N:! IntLiteral()] UInt(N) as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

// Conversions.

impl forall [To:! IntLiteral()] IntLiteral() as ImplicitAs(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert_checked";
}

final impl forall [From:! IntLiteral()] UInt(From) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl forall [To:! IntLiteral()] IntLiteral() as As(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert_checked";
}

final impl forall [From:! IntLiteral()] UInt(From) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Allow as an implicit conversion if To > From.
final impl forall [From:! IntLiteral(), To:! IntLiteral()] UInt(From) as As(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert";
}

// TODO: Allow as an implicit conversion if To > From.
final impl forall [From:! IntLiteral(), To:! IntLiteral()] UInt(From) as As(Int(To)) {
  fn Convert[self: Self]() -> Int(To) = "int.convert";
}

// Never implicit.
impl forall [From:! IntLiteral(), To:! IntLiteral()] Int(From) as As(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert";
}

// Comparisons.

// - UInt(N) vs UInt(M).

final impl forall [N:! IntLiteral(), M:! IntLiteral()] UInt(N) as EqWith(UInt(M)) {
  fn Equal[self: Self](other: UInt(M)) -> bool = "int.eq";
  fn NotEqual[self: Self](other: UInt(M)) -> bool = "int.neq";
}

final impl forall [N:! IntLiteral(), M:! IntLiteral()] UInt(N) as OrderedWith(UInt(M)) {
  // TODO: fn Compare
  fn Less[self: Self](other: UInt(M)) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: UInt(M)) -> bool = "int.less_eq";
  fn Greater[self: Self](other: UInt(M)) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: UInt(M)) -> bool = "int.greater_eq";
}

// - UInt(N) vs Int(M).

final impl forall [N:! IntLiteral(), M:! IntLiteral()] UInt(N) as EqWith(Int(M)) {
  fn Equal[self: Self](other: Int(M)) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Int(M)) -> bool = "int.neq";
}

final impl forall [N:! IntLiteral(), M:! IntLiteral()] UInt(N) as OrderedWith(Int(M)) {
  // TODO: fn Compare
  fn Less[self: Self](other: Int(M)) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Int(M)) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Int(M)) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Int(M)) -> bool = "int.greater_eq";
}

// - Int(N) vs UInt(M).

impl forall [N:! IntLiteral(), M:! IntLiteral()] Int(N) as EqWith(UInt(M)) {
  fn Equal[self: Self](other: UInt(M)) -> bool = "int.eq";
  fn NotEqual[self: Self](other: UInt(M)) -> bool = "int.neq";
}

impl forall [N:! IntLiteral(), M:! IntLiteral()] Int(N) as OrderedWith(UInt(M)) {
  // TODO: fn Compare
  fn Less[self: Self](other: UInt(M)) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: UInt(M)) -> bool = "int.less_eq";
  fn Greater[self: Self](other: UInt(M)) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: UInt(M)) -> bool = "int.greater_eq";
}

// - UInt(N) on left-hand side.

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))] UInt(N) as EqWith(T) {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))] UInt(N) as OrderedWith(T) {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

// - UInt(N) on right-hand side.

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))] T as EqWith(UInt(N)) {
  fn Equal[self: UInt(N)](other: UInt(N)) -> bool = "int.eq";
  fn NotEqual[self: UInt(N)](other: UInt(N)) -> bool = "int.neq";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))] T as OrderedWith(UInt(N)) {
  // TODO: fn Compare
  fn Less[self: UInt(N)](other: UInt(N)) -> bool = "int.less";
  fn LessOrEquivalent[self: UInt(N)](other: UInt(N)) -> bool = "int.less_eq";
  fn Greater[self: UInt(N)](other: UInt(N)) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: UInt(N)](other: UInt(N)) -> bool = "int.greater_eq";
}

// Arithmetic.

// - Homogeneous.

final impl forall [N:! IntLiteral()]
    UInt(N) as AddWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.uadd";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as DivWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.udiv";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as ModWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.umod";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as MulWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.umul";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as Negate where .Result = Self {
  fn Op[self: Self]() -> Self = "int.unegate";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as SubWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.usub";
}

// - UInt(N) on left-hand side.

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as AddWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as DivWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as ModWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as MulWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as SubWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
}

// - UInt(N) on right-hand side.

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as AddWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as DivWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as ModWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as MulWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as SubWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
}

// Bitwise operators.

// - Homogeneous.

final impl forall [N:! IntLiteral()]
    UInt(N) as BitAndWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as BitComplement where .Result = Self {
  fn Op[self: Self]() -> Self = "int.complement";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as BitOrWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as BitXorWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as LeftShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}

final impl forall [N:! IntLiteral()]
    UInt(N) as RightShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}

// - UInt(N) on left-hand side.

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as BitAndWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as BitOrWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as BitXorWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as LeftShiftWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as RightShiftWith(U) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
}

// - UInt(N) on right-hand side.

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as BitAndWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as BitOrWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as BitXorWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as LeftShiftWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
}

impl forall [N:! IntLiteral(), T:! ImplicitAs(UInt(N))]
    T as RightShiftWith(UInt(N)) where .Result = UInt(N) {
  fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
}

// Compound assignments.

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as AddAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.uadd_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as BitAndAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.and_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as BitOrAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.or_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as BitXorAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.xor_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as DivAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.udiv_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as LeftShiftAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.left_shift_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as ModAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.umod_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as MulAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.umul_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as RightShiftAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.right_shift_assign";
}

impl forall [N:! IntLiteral(), U:! ImplicitAs(UInt(N))]
    UInt(N) as SubAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.usub_assign";
}

// Increment and decrement.

// TODO: Add builtins for these to avoid emitting a call.

impl forall [N:! IntLiteral()] UInt(N) as Dec {
  fn Op[ref self: Self]() {
    // TODO: `self -= 1;` should work, but fails because the `impl IntLiteral
    // as ImplicitAs(Int(N))` is not final, which causes us to not attempt the
    // conversion from `1` to `Int(N)` until runtime, when we've lost the
    // constant value.
    fn AsUInt(n: IntLiteral()) -> UInt(N) = "int.convert_checked";
    self -= AsUInt(1);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as Inc {
  fn Op[ref self: Self]() {
    fn AsUInt(n: IntLiteral()) -> UInt(N) = "int.convert_checked";
    self += AsUInt(1);
  }
}
