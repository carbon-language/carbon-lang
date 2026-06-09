// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/float";

import library "prelude/copy";
import library "prelude/default";
import library "prelude/destroy";
import library "prelude/operators";
import library "prelude/types/float_literal";
import library "prelude/types/int";
import library "prelude/types/uint";
import library "prelude/types/int_literal";

private fn MakeFloat(size: IntLiteral) -> type = "float.make_type";

class Float(N:! IntLiteral) {
  adapt MakeFloat(N);
}

impl forall [N:! IntLiteral] Float(N) as UnformedInit {}

// Copy.

impl forall [N:! IntLiteral] Float(N) as Copy {
  fn Op(self) -> Self = "primitive_copy";
}

// Conversions between floating-point types. Implicit if all the source values
// can be represented exactly in the target type, otherwise explicit.
impl forall [N:! IntLiteral] FloatLiteral as ImplicitAs(Float(N)) {
  fn Convert(self) -> Float(N) = "float.convert_checked";
}

// TODO: Remove this once ImplicitAs extends As.
impl forall [N:! IntLiteral] FloatLiteral as As(Float(N)) {
  fn Convert(self) -> Float(N) = "float.convert_checked";
}

final impl forall [From:! IntLiteral, To:! IntLiteral]
    Float(From) as As(Float(To)) {
  fn Convert(self) -> Float(To) = "float.convert";
}

// AnyFloat exists to work around the inability to apply constraints to an impl
// that is only parameterized by integers.
// TODO: Remove these once possible.
private interface AnyFloat {
  let Width:! IntLiteral;
  fn AsFloat(self) -> Float(Width);
  fn Make(src: Float(Width)) -> Self;
}

impl forall [N:! IntLiteral] Float(N) as AnyFloat where .Width = N {
  fn AsFloat(self) -> Self { return self; }
  fn Make(src: Self) -> Self { return src; }
}

// Use `Float(?) as ImplicitAs(?)` to avoid a collision with the
// `? as ImplicitAs(Float(?))` type structure used by int -> float conversion.
// TODO: Use `match_first` to resolve this instead once it is available.
impl forall [From:! IntLiteral, To:! AnyFloat where Float(From) impls FloatFitsIn(.Self)]
    Float(From) as ImplicitAs(To) {
  fn Convert(self) -> To {
    fn Impl(src: Float(From)) -> Float(To.Width) = "float.convert";
    return To.Make(Impl(self));
  }
}

// Desired workaround-free version of the above:
// impl forall [From:! IntLiteral, To:! IntLiteral
//              where Float(From) impls FloatFitsIn(Float(To))]
//     Float(From) as ImplicitAs(Float(To)) {
//   fn Convert[self: Self]() -> Float(To) = "float.convert";
// }

// Conversions from floating-point to integer type. Always unsafe: not all
// source values fit in the target type.
final impl forall [From:! IntLiteral, To:! IntLiteral]
    Float(From) as UnsafeAs(Int(To)) {
  fn Convert(self) -> Int(To) = "float.convert_int";
}

final impl forall [From:! IntLiteral, To:! IntLiteral]
    Float(From) as UnsafeAs(UInt(To)) {
  fn Convert(self) -> UInt(To) = "float.convert_int";
}

final impl forall [From:! IntLiteral]
    Float(From) as UnsafeAs(IntLiteral) {
  fn Convert(self) -> IntLiteral = "float.convert_int";
}

// Conversions from integer to floating-point type. Implicit if all source
// values can be represented exactly in the target type, otherwise explicit.
impl forall [From:! IntLiteral, To:! IntLiteral]
    Int(From) as As(Float(To)) {
  fn Convert(self) -> Float(To) = "int.convert_float";
}

impl forall [From:! IntLiteral, To:! IntLiteral]
    UInt(From) as As(Float(To)) {
  fn Convert(self) -> Float(To) = "int.convert_float";
}

impl forall [To:! IntLiteral]
    IntLiteral as As(Float(To)) {
  fn Convert(self) -> Float(To) = "int.convert_float";
}

// AnyInt, AnyUInt, and IntImplicitAsFloat exist to work around the inability to
// apply constraints to an impl that is only parameterized by integers, and the
// lack of support for `match_first`.
// TODO: Remove these once possible.
private interface AnyInt {
  let Width:! IntLiteral;
  fn AsInt(self) -> Int(Width);
}

impl forall [N:! IntLiteral] Int(N) as AnyInt where .Width = N {
  fn AsInt(self) -> Self { return self; }
}

private interface AnyUInt {
  let Width:! IntLiteral;
  fn AsUInt(self) -> UInt(Width);
}

impl forall [N:! IntLiteral] UInt(N) as AnyUInt where .Width = N {
  fn AsUInt(self) -> Self { return self; }
}

private interface IntImplicitAsFloat(To:! IntLiteral) {
  fn Convert(self) -> Float(To);
}

final impl forall [To:! IntLiteral, From:! AnyInt & IntFitsIn(Float(To))]
    From as IntImplicitAsFloat(To) {
  fn Convert(self) -> Float(To) {
    fn Impl(src: Int(From.Width)) -> Float(To) = "int.convert_float";
    return Impl(self.AsInt());
  }
}

impl forall [To:! IntLiteral, From:! AnyUInt & IntFitsIn(Float(To))]
    From as IntImplicitAsFloat(To) {
  fn Convert(self) -> Float(To) {
    fn Impl(src: UInt(From.Width)) -> Float(To) = "int.convert_float";
    return Impl(self.AsUInt());
  }
}

impl forall [To:! IntLiteral, From:! IntImplicitAsFloat(To)]
    From as ImplicitAs(Float(To)) {
  fn Convert(self) -> Float(To) {
    return self.Convert();
  }
}

// Desired workaround-free version of the above:
//
// impl forall [From:! IntLiteral, To:! IntLiteral
//              where Int(From) impls ImplicitAs(Float(To))]
//     Int(From) as ImplicitAs(Float(To)) {
//   fn Convert[self: Self]() -> Float(To) = "int.convert_float";
// }
//
// impl forall [From:! IntLiteral, To:! IntLiteral
//              where UInt(From) impls ImplicitAs(Float(To))]
//     UInt(From) as ImplicitAs(Float(To)) {
//   fn Convert[self: Self]() -> Float(To) = "int.convert_float";
// }

impl forall [To:! IntLiteral]
    IntLiteral as ImplicitAs(Float(To)) {
  fn Convert(self) -> Float(To) = "int.convert_float_checked";
}

// Comparisons.
// TODO: Support mixed-type comparisons.
final impl forall [N:! IntLiteral] Float(N) as EqWith(Float(N)) {
  fn Equal(self, other: Float(N)) -> bool = "float.eq";
  fn NotEqual(self, other: Float(N)) -> bool = "float.neq";
}

final impl forall [N:! IntLiteral] Float(N) as OrderedWith(Float(N)) {
  fn Less(self, other: Float(N)) -> bool = "float.less";
  fn LessOrEquivalent(self, other: Float(N)) -> bool = "float.less_eq";
  fn Greater(self, other: Float(N)) -> bool = "float.greater";
  fn GreaterOrEquivalent(self, other: Float(N)) -> bool = "float.greater_eq";
}

// Arithmetic.
final impl forall [N:! IntLiteral]
    Float(N) as AddWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "float.add";
}

final impl forall [N:! IntLiteral]
    Float(N) as DivWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "float.div";
}

final impl forall [N:! IntLiteral]
    Float(N) as MulWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "float.mul";
}

final impl forall [N:! IntLiteral]
    Float(N) as Negate where .Result = Self {
  fn Op(self) -> Self = "float.negate";
}

final impl forall [N:! IntLiteral]
    Float(N) as SubWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "float.sub";
}

// Compound assignments.
final impl forall [N:! IntLiteral]
    Float(N) as AddAssignWith(Self) {
  fn Op(ref self, other: Self) = "float.add_assign";
}

final impl forall [N:! IntLiteral]
    Float(N) as DivAssignWith(Self) {
  fn Op(ref self, other: Self) = "float.div_assign";
}

final impl forall [N:! IntLiteral]
    Float(N) as MulAssignWith(Self) {
  fn Op(ref self, other: Self) = "float.mul_assign";
}

final impl forall [N:! IntLiteral]
    Float(N) as SubAssignWith(Self) {
  fn Op(ref self, other: Self) = "float.sub_assign";
}
