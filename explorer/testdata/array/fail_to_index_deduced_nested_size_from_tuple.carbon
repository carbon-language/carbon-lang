// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var x: [[i32;];] = ((1,2), (3,4));
  // CHECK:STDERR: RUNTIME ERROR: fail_to_index_deduced_nested_size_from_tuple.carbon:[[@LINE+1]]: index 2 out of range in (1, 2)
  return x[0][2];
}
