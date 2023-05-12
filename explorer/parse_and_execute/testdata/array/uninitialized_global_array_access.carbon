// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 100
// CHECK:STDOUT: result: 0

package ExplorerTest api;

var my_array : [i32; 1];

fn Main() -> i32 {
  my_array[0] = 100;
  Print("{0}", my_array[0]);
  return 0;
}
