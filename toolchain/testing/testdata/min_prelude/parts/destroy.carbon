// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/destroy.carbon

package Core library "prelude/parts/destroy";

interface Destroy {
  fn Op[addr self: Self*]();
}
impl forall [T:! type] T as Destroy {
  fn Op[addr self: Self*]() = "no_op";
}
