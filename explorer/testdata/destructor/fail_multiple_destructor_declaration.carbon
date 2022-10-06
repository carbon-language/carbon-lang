// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

class A{
    destructor[me: Self]{
        Print("DESTRUCTOR A1 {0}",me.n);
    }
    destructor[me: Self]{
        Print("DESTRUCTOR A2 {0}",me.n);
    // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/destructor/fail_multiple_destructor_declaration.carbon:[[@LINE+1]]: Duplicate name `destructor` also found at {{.*}}/explorer/testdata/destructor/fail_multiple_destructor_declaration.carbon:14
    }
    var n: i32;
}

fn Main() -> i32 {
  var a: A = {.n = 3};
  return 1;
}
