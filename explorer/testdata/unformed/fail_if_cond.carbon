// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var v: bool;
  // CHECK:STDERR: COMPILATION ERROR: fail_if_cond.carbon:[[@LINE+1]]: use of uninitialized variable v
  return if v then 1 else 2;
}
