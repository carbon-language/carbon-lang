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
  extend base: C;
// CHECK:STDERR: COMPILATION ERROR: fail_extend_after_var.carbon:[[@LINE+1]]: `extend base:` declaration must not be after `var` or `mix` declarations in a class.
}

fn Main() -> i32 {
  return 0;
}
