// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn F() {}
fn G() {}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/basic_syntax/fail_assign_to_function.carbon:[[@LINE+1]]: Only a reference expression can be assigned to, but got `F`
  F = G;
  return 0;
}
