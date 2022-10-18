// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 6

package ExplorerTest api;

// The constraint here resolves to:
//
//   MulWith(T) where <ConstraintSelf>.(MulWith(T).Result) == <ConstraintSelf>
//
// Note in particular that this involves a member of `MulWith(T)`, so we need
// `T` to have a symbolic identity when checking its own type.
fn DoMul[T:! MulWith(.Self) where .Result == .Self](x: T, y: T) -> T {
  return x * y;
}

fn Main() -> i32 {
  return DoMul(2, 3);
}
