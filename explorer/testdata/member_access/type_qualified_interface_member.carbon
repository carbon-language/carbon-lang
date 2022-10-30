// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 42

package Foo api;
interface HasDefault {
  fn Default() -> Self;
}
impl i32 as HasDefault {
  fn Default() -> i32 { return 42; }
}
fn Main() -> i32 {
  return i32.(HasDefault.Default)();
}
