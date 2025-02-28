// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/as";

interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface ImplicitAs(Dest:! type) {
  // TODO: extend As(Dest);
  fn Convert[self: Self]() -> Dest;
}
