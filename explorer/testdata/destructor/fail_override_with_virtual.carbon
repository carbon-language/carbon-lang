// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;


base class A{
    virtual destructor[self: Self]{}
}

class B extends A {
    virtual destructor[self: Self]{}
}

fn Main() -> i32 {
  return 0;
}
