// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/float";

import library "prelude/copy";
import library "prelude/default";
import library "prelude/destroy";
import library "prelude/operators";
import library "prelude/types/float_literal";
import library "prelude/types/int_literal";

private fn MakeFloat(size: IntLiteral()) -> type = "float.make_type";

class Float(N:! IntLiteral()) {
  adapt MakeFloat(N);
}

impl forall [N:! IntLiteral()] Float(N) as UnformedInit {}

// Copy.

impl forall [N:! IntLiteral()] Float(N) as Copy {
  fn Op(self) -> Self = "primitive_copy";
}

// Conversions.
// TODO: Support mixed-size conversions.
// TODO: Support int-to-float conversions.
impl forall [N:! IntLiteral()] FloatLiteral() as ImplicitAs(Float(N)) {
  fn Convert(self) -> Float(N) = "float.convert_checked";
}

// TODO: Remove this once ImplicitAs extends As.
impl forall [N:! IntLiteral()] FloatLiteral() as As(Float(N)) {
  fn Convert(self) -> Float(N) = "float.convert_checked";
}

// Comparisons.
// TODO: Support mixed-type comparisons.
final impl forall [N:! IntLiteral()] Float(N) as EqWith(Float(N)) {
  fn Equal(self, other: Float(N)) -> bool = "float.eq";
  fn NotEqual(self, other: Float(N)) -> bool = "float.neq";
}

final impl forall [N:! IntLiteral()] Float(N) as OrderedWith(Float(N)) {
  fn Less(self, other: Float(N)) -> bool = "float.less";
  fn LessOrEquivalent(self, other: Float(N)) -> bool = "float.less_eq";
  fn Greater(self, other: Float(N)) -> bool = "float.greater";
  fn GreaterOrEquivalent(self, other: Float(N)) -> bool = "float.greater_eq";
}

// Arithmetic.
final impl forall [N:! IntLiteral()]
    Float(N) as AddWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "float.add";
}

final impl forall [N:! IntLiteral()]
    Float(N) as DivWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "float.div";
}

final impl forall [N:! IntLiteral()]
    Float(N) as MulWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "float.mul";
}

final impl forall [N:! IntLiteral()]
    Float(N) as Negate where .Result = Self {
  fn Op(self) -> Self = "float.negate";
}

final impl forall [N:! IntLiteral()]
    Float(N) as SubWith(Self) where .Result = Self {
  fn Op(self, other: Self) -> Self = "float.sub";
}

// Compound assignments.
final impl forall [N:! IntLiteral()]
    Float(N) as AddAssignWith(Self) {
  fn Op(ref self, other: Self) = "float.add_assign";
}

final impl forall [N:! IntLiteral()]
    Float(N) as DivAssignWith(Self) {
  fn Op(ref self, other: Self) = "float.div_assign";
}

final impl forall [N:! IntLiteral()]
    Float(N) as MulAssignWith(Self) {
  fn Op(ref self, other: Self) = "float.mul_assign";
}

final impl forall [N:! IntLiteral()]
    Float(N) as SubAssignWith(Self) {
  fn Op(ref self, other: Self) = "float.sub_assign";
}
