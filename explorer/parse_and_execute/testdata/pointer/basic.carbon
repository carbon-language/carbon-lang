// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32 = 5;
  // changes x to 10
  x = 10;
  var p: i32* = &x;
  // changes x to 7
  *p = 7;
  var q: i32* = &*p;
  // changes x to 0
  *q = 0;
  var y: i32 = *p;

  return y;
}
