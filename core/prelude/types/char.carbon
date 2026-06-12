// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/char";

import library "prelude/copy";
import library "prelude/default";
import library "prelude/destroy";
import library "prelude/operators";
import library "prelude/types/uint";
import library "prelude/types/char_literal";
import library "prelude/types/int";
import library "prelude/types/int_literal";

class Char {
  adapt u8;
}

impl Char as UnformedInit {}

impl Char as Copy {
  fn Op(self) -> Self = "primitive_copy";
}

// Conversions.

impl CharLiteral as ImplicitAs(Char) {
  fn Convert(self) -> Char = "char_literal.convert_char";
}

// TODO: Remove these once ImplicitAs extends As.
impl CharLiteral as As(Char) {
  fn Convert(self) -> Char = "char_literal.convert_char";
}

impl forall [From:! IntLiteral] UInt(From) as As(Char) {
  fn Convert(self) -> Char = "int.convert_char";
}
// `char` explicitly converts to any type that `u8` implicitly converts to.
impl forall [T:! type where u8 impls ImplicitAs(.Self)] Char as As(T) {
  fn Convert(self) -> T {
    return self as u8;
  }
}

// Comparisons.

impl Char as EqWith(Char) {
  fn Equal(self, other: Char) -> bool = "int.eq";
  fn NotEqual(self, other: Char) -> bool = "int.neq";
}

impl Char as OrderedWith(Char) {
  fn Less(self, other: Char) -> bool = "int.less";
  fn LessOrEquivalent(self, other: Char) -> bool = "int.less_eq";
  fn Greater(self, other: Char) -> bool = "int.greater";
  fn GreaterOrEquivalent(self, other: Char) -> bool = "int.greater_eq";
}

impl forall [T:! ImplicitAs(Char)] Char as EqWith(T) {
  fn Equal(self, other: Char) -> bool = "int.eq";
  fn NotEqual(self, other: Char) -> bool = "int.neq";
}

impl forall [T:! ImplicitAs(Char)] Char as OrderedWith(T) {
  fn Less(self, other: Char) -> bool = "int.less";
  fn LessOrEquivalent(self, other: Char) -> bool = "int.less_eq";
  fn Greater(self, other: Char) -> bool = "int.greater";
  fn GreaterOrEquivalent(self, other: Char) -> bool = "int.greater_eq";
}

impl forall [T:! ImplicitAs(Char)] T as EqWith(Char) {
  fn Equal(self: Char, other: Char) -> bool = "int.eq";
  fn NotEqual(self: Char, other: Char) -> bool = "int.neq";
}

impl forall [T:! ImplicitAs(Char)] T as OrderedWith(Char) {
  fn Less(self: Char, other: Char) -> bool = "int.less";
  fn LessOrEquivalent(self: Char, other: Char) -> bool = "int.less_eq";
  fn Greater(self: Char, other: Char) -> bool = "int.greater";
  fn GreaterOrEquivalent(self: Char, other: Char) -> bool = "int.greater_eq";
}

// Arithmetic.

private interface AnyInt {}

impl forall [N:! IntLiteral] Int(N) as AnyInt {}
impl forall [N:! IntLiteral] UInt(N) as AnyInt {}

impl forall [T:! AnyInt & As(u8)]
    Char as AddWith(T) where .Result = Char {
  fn Op(self, other: T) -> Self {
    // TODO: This should be erroneous if the addition overflows.
    return ((self as u8) + (other as u8)) as Char;
  }
}

impl forall [T:! AnyInt & As(u8)]
    T as AddWith(Char) where .Result = Char {
  fn Op(self, other: Char) -> Char {
    // TODO: This should be erroneous if the addition overflows.
    return ((self as u8) + (other as u8)) as Char;
  }
}

final impl forall [T:! AnyInt & As(u8)]
    Char as SubWith(T) where .Result = Char {
  fn Op(self, other: T) -> Self {
    // TODO: This should be erroneous if the addition overflows.
    return ((self as u8) - (other as u8)) as Char;
  }
}

impl Char as SubWith(Char) where .Result = i32 {
  fn Op(self, other: Char) -> i32 {
    return (self as i32) - (other as i32);
  }
}

impl forall [T:! ImplicitAs(Char)] T as SubWith(Char) where .Result = i32 {
  fn Op(self: Char, other: Char) -> i32 {
    return (self as i32) - (other as i32);
  }
}

impl forall [T:! ImplicitAs(Char)] Char as SubWith(T) where .Result = i32 {
  fn Op(self, other: Char) -> i32 {
    return (self as i32) - (other as i32);
  }
}
