// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A{
    destructor[self: Self]{
        Print("DESTRUCTOR A1 {0}",self.n);
    }
    destructor[self: Self]{
        Print("DESTRUCTOR A2 {0}",self.n);
    // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/destructor/fail_multiple_destructor_declaration.carbon:[[@LINE+1]]: Duplicate name `destructor` also found at explorer/testdata/destructor/fail_multiple_destructor_declaration.carbon:[[@LINE-3]]
    }
    var n: i32;
}

fn Main() -> i32 {
  var a: A = {.n = 3};
  return 1;
}
