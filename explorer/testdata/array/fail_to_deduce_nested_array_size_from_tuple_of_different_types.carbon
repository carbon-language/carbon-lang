// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: fail_to_deduce_nested_array_size_from_tuple_of_different_types.carbon:[[@LINE+1]]: type pattern '{{\[\[}}i32;]; 2]' does not match actual type '((i32, i32), (String, String))'
  var x: [[i32;];2] = ((1,2), ("foo", "bar"));
  return x[0];
}
