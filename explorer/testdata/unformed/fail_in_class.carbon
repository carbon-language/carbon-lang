// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  fn F() -> i32 {
    var n: i32;
    // CHECK:STDERR: COMPILATION ERROR: fail_in_class.carbon:[[@LINE+1]]: use of uninitialized variable n
    return n;
  }
}

fn Main() -> i32 {
  return C.F();
}
