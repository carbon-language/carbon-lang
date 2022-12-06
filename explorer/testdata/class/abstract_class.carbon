// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: d.a=1
// CHECK:STDOUT: d.b=2
// CHECK:STDOUT: result: 0

package ExplorerTest api;

abstract class C {
  var a: i32;
}

class D extends C {
  var b: i32;
}

fn Main() -> i32 {
  var d: D = { .base = {.a = 1}, .b = 2 };
  Print("d.a={0}", d.a);
  Print("d.b={0}", d.b);
  return 0;
}
