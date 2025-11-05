// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/as";

import library "prelude/copy";

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

impl forall [T:! Copy] T as ImplicitAs(T) {
  fn Convert[self: Self]() -> Self { return self.(Copy.Op)(); }
}

impl forall [T:! type, U:! ImplicitAs(T)] U as ImplicitAs(const T) {
  fn Convert[self: U]() -> const T { return self.Convert(); }
}

impl forall [T:! type, U:! ImplicitAs(T)] const U as ImplicitAs(T) {
  fn Convert[self: const U]() -> T { return (self as U).Convert(); }
}
