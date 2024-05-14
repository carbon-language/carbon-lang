// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class E {
  base class C {
  }

  // This should be allowed, but Explorer forbids it until the problem from
  // https://github.com/carbon-language/carbon-lang/issues/2994 can be fixed.

  // CHECK:STDERR: COMPILATION ERROR: fail_extend_member_class.carbon:[[@LINE+1]]: `extend base:` declarations after the first declaration in the class are not yet supported
  extend base: C;
}

fn Main() -> i32 {
  return 0;
}
