// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/float";

import library "prelude/destroy";
import library "prelude/operators";
import library "prelude/types/float_literal";
import library "prelude/types/int_literal";

private fn MakeFloat(size: IntLiteral()) -> type = "float.make_type";

fn ConvertFromLegacyFloat[T:! type](literal: FloatLiteral()) -> T =
    "float.from_legacy_float";

class Float(N:! IntLiteral()) {
  adapt MakeFloat(N);
}

// Comparisons.
final impl forall [N:! IntLiteral(), M:! IntLiteral()] Float(N) as EqWith(Float(M)) {
  fn Equal[self: Self](other: Float(M)) -> bool = "float.eq";
  fn NotEqual[self: Self](other: Float(M)) -> bool = "float.neq";
}

final impl forall [N:! IntLiteral(), M:! IntLiteral()] Float(N) as OrderedWith(Float(M)) {
  fn Less[self: Self](other: Float(M)) -> bool = "float.less";
  fn LessOrEquivalent[self: Self](other: Float(M)) -> bool = "float.less_eq";
  fn Greater[self: Self](other: Float(M)) -> bool = "float.greater";
  fn GreaterOrEquivalent[self: Self](other: Float(M)) -> bool = "float.greater_eq";
}

// Arithmetic.
final impl forall [N:! IntLiteral()]
    Float(N) as AddWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "float.add";
}

final impl forall [N:! IntLiteral()]
    Float(N) as DivWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "float.div";
}

final impl forall [N:! IntLiteral()]
    Float(N) as MulWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "float.mul";
}

final impl forall [N:! IntLiteral()]
    Float(N) as Negate where .Result = Self {
  fn Op[self: Self]() -> Self = "float.negate";
}

final impl forall [N:! IntLiteral()]
    Float(N) as SubWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "float.sub";
}

// Compound assignments.
impl forall [N:! IntLiteral()]
    Float(N) as AddAssignWith(Self) {
  fn Op[addr self: Self*](other: Self) = "float.add_assign";
}

impl forall [N:! IntLiteral()]
    Float(N) as DivAssignWith(Self) {
  fn Op[addr self: Self*](other: Self) = "float.div_assign";
}

impl forall [N:! IntLiteral()]
    Float(N) as MulAssignWith(Self) {
  fn Op[addr self: Self*](other: Self) = "float.mul_assign";
}

impl forall [N:! IntLiteral()]
    Float(N) as SubAssignWith(Self) {
  fn Op[addr self: Self*](other: Self) = "float.sub_assign";
}
