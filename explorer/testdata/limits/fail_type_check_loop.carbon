// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest impl;

fn Loop() -> type {
  // CHECK:STDERR: COMPILATION ERROR: fail_type_check_loop.carbon:[[@LINE+1]]: possible infinite loop: too many interpreter steps executed
  while (true) {}
  return i32;
}

fn Main() -> Loop() {
  return 0;
}
