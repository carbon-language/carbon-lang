// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: fail_to_deduce_size_no_expression_to_deduce_from.carbon:[[@LINE+1]]: cannot deduce size for x: [i32;]
  var x: [i32;];
  return x[0];
}
