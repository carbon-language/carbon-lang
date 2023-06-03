// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: fail_to_deduce_size_from_int.carbon:[[@LINE+1]]: type pattern '[i32;]' does not match actual type 'i32'
  var x: [i32;] = 42;
  return x[0];
}
