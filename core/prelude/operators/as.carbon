// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/as";

// For Core.Int, used below.
// TODO: Remove this when the below `impl`s are removed.
import library "prelude/types";

interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface ImplicitAs(Dest:! type) {
  // TODO: extend As(Dest);
  fn Convert[self: Self]() -> Dest;
}

// TODO: These impls should live with Core.Int, but currently that's a builtin
// not a class type, so there is no other library these can go in.
impl forall [N:! IntLiteral()] IntLiteral() as ImplicitAs(Int(N)) {
  fn Convert[self: Self]() -> Int(N) = "int.convert_checked";
}

impl forall [N:! IntLiteral()] IntLiteral() as ImplicitAs(UInt(N)) {
  fn Convert[self: Self]() -> UInt(N) = "int.convert_checked";
}

impl forall [N:! IntLiteral()] Int(N) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

impl forall [N:! IntLiteral()] UInt(N) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl forall [N:! IntLiteral()] IntLiteral() as As(Int(N)) {
  fn Convert[self: Self]() -> Int(N) = "int.convert_checked";
}

impl forall [N:! IntLiteral()] IntLiteral() as As(UInt(N)) {
  fn Convert[self: Self]() -> UInt(N) = "int.convert_checked";
}

impl forall [N:! IntLiteral()] Int(N) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

impl forall [N:! IntLiteral()] UInt(N) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}
