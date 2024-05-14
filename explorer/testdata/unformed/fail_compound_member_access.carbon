// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  var n: i32;
}

fn Main() -> i32 {
  var c: C;
  // CHECK:STDERR: COMPILATION ERROR: fail_compound_member_access.carbon:[[@LINE+1]]: use of uninitialized variable c
  return c.(C.n);
}
