// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var v: i32;
  // CHECK:STDERR: COMPILATION ERROR: fail_if_then.carbon:[[@LINE+1]]: use of uninitialized variable v
  return if true then v else 2;
}
