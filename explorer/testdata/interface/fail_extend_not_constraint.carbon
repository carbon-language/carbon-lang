// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Vector {
  // CHECK:STDERR: COMPILATION ERROR: fail_extend_not_constraint.carbon:[[@LINE+1]]: expected a constraint in extend declaration, found i32
  extend i32;
}

fn Main() -> i32 {
  return 0;
}
