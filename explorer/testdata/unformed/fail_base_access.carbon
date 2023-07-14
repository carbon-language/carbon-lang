// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

base class B {
  var n: i32;
}

class D {
  extend base: B;
}

fn Main() -> i32 {
  var d: D;
  // CHECK:STDERR: COMPILATION ERROR: fail_base_access.carbon:[[@LINE+1]]: use of uninitialized variable d
  return d.n;
}
