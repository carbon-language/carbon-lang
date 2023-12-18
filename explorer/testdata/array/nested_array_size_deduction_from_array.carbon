// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var arr: [[i32;2];2] = ((1,2), (3,4));

  var x : [[i32;];] = arr;
  x[0][1] = 1;
  x[1][0] = 1;
  return x[0][0] + x[0][1] + x[1][0] + x[1][1];
}

// CHECK:STDOUT: result: 7
