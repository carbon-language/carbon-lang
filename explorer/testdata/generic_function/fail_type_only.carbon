// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {}

// CHECK:STDERR: SYNTAX ERROR: explorer/testdata/generic_function/fail_type_only.carbon:[[@LINE+1]]: syntax error, unexpected RIGHT_SQUARE_BRACKET, expecting COLON
fn f[x:! i32, addr C]() {}

fn Main() -> i32 {
  return 0;
}
