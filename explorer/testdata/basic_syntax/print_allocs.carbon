// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var a: i32 = 1;
  var b: i32 = 2;
  heap.PrintAllocs();
  return 0;
}

// CHECK:STDOUT: 0: Heap{}, 1: 1, 2: 2
// CHECK:STDOUT: result: 0
