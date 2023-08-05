// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

base class C {
}

class E {
  var x: i32;
  // CHECK:STDERR: COMPILATION ERROR: fail_extend_after_var.carbon:[[@LINE+1]]: `extend base:` declarations after the first declaration in the class are not yet supported
  extend base: C;
}

fn Main() -> i32 {
  return 0;
}
