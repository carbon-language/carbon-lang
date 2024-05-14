// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var arr: [i32; 2] = (0, 1);
  var x: [i32;] = arr;
  // CHECK:STDERR: RUNTIME ERROR: fail_index_deduced_size_from_array.carbon:[[@LINE+1]]: index 2 out of range in (0, 1)
  return x[2];
}
