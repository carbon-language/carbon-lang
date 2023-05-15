// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Vector {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/interface/fail_qualified_method.carbon:[[@LINE+1]]: qualified declaration names are not permitted in this context
  fn Vector.Zero() -> i32;
}

fn Main() -> i32 {
  return 0;
}
