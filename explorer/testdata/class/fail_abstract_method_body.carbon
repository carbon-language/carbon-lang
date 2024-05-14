// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

abstract class A {
    // CHECK:STDERR: COMPILATION ERROR: fail_abstract_method_body.carbon:[[@LINE+1]]: Error declaring `Foo`: abstract method cannot have a body.
    abstract fn Foo[self: Self]() {}
}


fn Main() -> i32 {
  return 0;
}
