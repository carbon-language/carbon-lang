// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;


base class A {
    // CHECK:STDERR: SYNTAX ERROR: explorer/testdata/destructor/fail_override_abstract.carbon:[[@LINE+1]]: syntax error, unexpected DESTRUCTOR, expecting CLASS
    abstract destructor[self: Self]{}
}

fn Main() -> i32 {
  return 0;
}
