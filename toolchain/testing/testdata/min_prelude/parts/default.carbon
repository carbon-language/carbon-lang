// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/default.carbon

package Core library "prelude/parts/default";

interface DefaultOrUnformed { fn Op() -> Self; }

// In tests, just allow any type to be default-initialized as a no-op.
// TODO: This is a hack to allow existing tests to contiue working.
impl forall [T:! type] T as DefaultOrUnformed {
  fn Op() -> Self = "make_uninitialized";
}
