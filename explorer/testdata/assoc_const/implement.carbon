// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 2

package ExplorerTest api;

interface Vector {
  let Dim:! i32;
}

class Point {
  var x: i32;
  var y: i32;
  impl as Vector where .Dim == 2 {}
}

fn Main() -> i32 {
  return Point.(Vector.Dim);
}
