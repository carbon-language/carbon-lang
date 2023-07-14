// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

base class C {
}

base class D {
}

class E {
  extend base: C;
  // CHECK:STDERR: COMPILATION ERROR: fail_extend_twice.carbon:[[@LINE+1]]: At most one `extend base:` declaration in a class.
  extend base: D;
}

fn Main() -> i32 {
  return 0;
}
