// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class A {
    // CHECK:STDERR: COMPILATION ERROR: fail_abstract_method_non_abstract_class.carbon:[[@LINE+1]]: Error declaring `Foo`: `abstract` methods are allowed only in abstract classes.
    abstract fn Foo[self: Self]();
}


fn Main() -> i32 {
  return 0;
}
