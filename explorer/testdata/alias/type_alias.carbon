// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 4

package ExplorerTest api;

alias TypeAlias = i32;

fn Main() -> TypeAlias {
  var n: TypeAlias = 1;
  var m: i32 = n;
  var p: i32* = &n;
  var q: TypeAlias* = &m;
  return *p + *q + m + n;
}
