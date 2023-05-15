// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn F() -> () {
// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/return/fail_explicit_with_no_return.carbon:[[@LINE+1]]: control-flow reaches end of function that provides a `->` return type without reaching a return statement
}

fn Main() -> i32 {
  F();
  return 0;
}
