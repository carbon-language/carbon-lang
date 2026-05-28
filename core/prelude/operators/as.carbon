// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/as";

import library "prelude/copy";
import library "prelude/types/int_literal";
import library "prelude/types/float_literal";

interface UnsafeAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface As(Dest:! type) {
  // TODO: extend UnsafeAs(Dest);
  fn Convert[self: Self]() -> Dest;
}

interface ImplicitAs(Dest:! type) {
  // TODO: extend As(Dest);
  fn Convert[self: Self]() -> Dest;
}

// Workaround: ImplicitAs extends As.
impl forall [U:! type, T:! ImplicitAs(U)] T as As(U) {
  fn Convert[self: Self]() -> U { return self.Convert(); }
}

// Workaround: As extends UnsafeAs.
impl forall [U:! type, T:! As(U)] T as UnsafeAs(U) {
  fn Convert[self: Self]() -> U { return self.Convert(); }
}

// Copyable types have an identity conversion that performs a copy.
impl forall [T:! Copy] T as ImplicitAs(T) {
  fn Convert[self: Self]() -> Self { return self.(Copy.Op)(); }
}

// `const` can be added and removed when converting a value.
impl forall [T:! type, U:! ImplicitAs(T)] U as ImplicitAs(const T) {
  fn Convert[self: U]() -> const T { return self.Convert(); }
}

impl forall [T:! type, U:! ImplicitAs(T)] const U as ImplicitAs(T) {
  fn Convert[self: const U]() -> T { return (self as U).Convert(); }
}

// `const` can be added to a pointer.
// TODO: This is also provided as a builtin conversion. We provide it here so
// that Optional(T*) can implicitly convert to Optional(const T*). See #5750.
impl forall [T:! type] T* as ImplicitAs(const T*) {
  fn Convert[self: T*]() -> const T* = "pointer.unsafe_convert";
}

impl forall [T:! type] T* as As(const T*) {
  fn Convert[self: T*]() -> const T* = "pointer.unsafe_convert";
}

// Pointer types can be unsafely cast to other pointer types.
// TODO: Should `unsafe as` be able to remove `const`?
impl forall [T:! type, U:! type] T* as UnsafeAs(U*) {
  fn Convert[self: T*]() -> U* = "pointer.unsafe_convert";
}

// Integer literals can be implicitly converted to floating-point literals.
impl IntLiteral() as ImplicitAs(FloatLiteral()) {
  fn Convert[self: Self]() -> FloatLiteral() = "int.convert_float_checked";
}

// Floating-point literals can be explicitly cast to integer literals.
impl FloatLiteral() as UnsafeAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "float.convert_int";
}

interface IntFitsIn(Dest:! type) {}
interface FloatFitsIn(Src:! type) {}
