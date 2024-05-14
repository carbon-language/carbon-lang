// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var x: [i32; 2] = (1, 2);
  var y: [[i32;];] = (x, (3,4));
  y[0][1] = 1;
  y[1][0] = 1;
  return y[0][0] + y[0][1] + y[1][0] + y[1][1];
}

// CHECK:STDOUT: result: 7
