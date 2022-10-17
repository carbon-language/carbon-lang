// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A { fn F[me: Self]() -> i32 { return 0; } }

fn Main() -> i32 {
  // {} is a bit strange; its type is {}. Make sure we can properly convert a
  // value of type {} to type TypeOfClassType.
  var a: A = {};
  return a.F();
}
