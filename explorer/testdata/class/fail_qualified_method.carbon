// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class Point {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/class/fail_qualified_method.carbon:[[@LINE+1]]: qualified declaration names are not permitted in this context
  fn Point.F[self: Self]() {}
}

fn Main() -> i32 {
  return 0;
}
